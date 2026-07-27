import pytest
import torch.distributed as dist
from pytest_lazy_fixtures import lf
from torch.utils.data import DataLoader

from rbyte import Dataset
from rbyte.dataloader import TorchDataNodeDataLoader, collate_identity


@pytest.fixture
def common_kwargs() -> dict[str, object]:
    return {
        "batch_size": 2,
        "shuffle": False,
        "collate_fn": collate_identity,
        "num_workers": 2,
    }


@pytest.fixture
def torch_dataloader(dataset: Dataset, common_kwargs: dict[str, object]) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        multiprocessing_context="forkserver",
        **common_kwargs,  # ty:ignore[invalid-argument-type]
    )


@pytest.fixture(params=[pytest.param("process"), pytest.param("thread")])
def torchdata_dataloader(
    dataset: Dataset, request: pytest.FixtureRequest, common_kwargs: dict[str, object]
) -> TorchDataNodeDataLoader:
    match request.param:
        case "process":
            kwargs = {"multiprocessing_context": "forkserver", "method": "process"}

        case "thread":
            kwargs = {"method": "thread"}

        case _:
            raise RuntimeError

    return TorchDataNodeDataLoader(
        dataset=dataset,
        **(common_kwargs | kwargs),  # ty:ignore[invalid-argument-type]
    )


@pytest.mark.parametrize("dataset", [lf("yaak_dataset")])
def test_dataloaders(
    torch_dataloader: DataLoader, torchdata_dataloader: TorchDataNodeDataLoader
) -> None:
    for left, right in zip(torch_dataloader, torchdata_dataloader, strict=True):
        assert (left == right).all()


EXPECTED_EPOCH = 3


@pytest.fixture
def _gloo_process_group():  # noqa: ANN202
    dist.init_process_group(
        backend="gloo", init_method="tcp://127.0.0.1:29517", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


def test_no_sharding_without_process_group(
    yaak_dataset: Dataset, common_kwargs: dict[str, object]
) -> None:
    loader = TorchDataNodeDataLoader(
        dataset=yaak_dataset,
        method="thread",
        **common_kwargs,  # ty:ignore[invalid-argument-type]
    )
    assert len(loader) > 0
    assert loader._distributed_sampler is None  # noqa: SLF001


@pytest.mark.usefixtures("_gloo_process_group")
def test_shards_with_process_group(
    yaak_dataset: Dataset, common_kwargs: dict[str, object]
) -> None:
    loader = TorchDataNodeDataLoader(
        dataset=yaak_dataset,
        method="thread",
        **common_kwargs,  # ty:ignore[invalid-argument-type]
    )
    assert len(loader) > 0  # triggers the lazy build
    sampler = loader._distributed_sampler  # noqa: SLF001
    assert sampler is not None
    assert sampler.num_replicas == 1
    loader.set_epoch(EXPECTED_EPOCH)
    assert sampler.epoch == EXPECTED_EPOCH


@pytest.mark.usefixtures("_gloo_process_group")
def test_rebuilds_if_group_appears_late(
    yaak_dataset: Dataset, common_kwargs: dict[str, object]
) -> None:
    loader = TorchDataNodeDataLoader(
        dataset=yaak_dataset,
        method="thread",
        **common_kwargs,  # ty:ignore[invalid-argument-type]
    )
    # simulate a build that happened before the process group existed
    loader._built_distributed = False  # noqa: SLF001
    loader._loader = object()  # noqa: SLF001  # ty: ignore[invalid-assignment]
    assert len(loader) > 0
    assert loader._distributed_sampler is not None  # noqa: SLF001


@pytest.mark.usefixtures("_gloo_process_group")
def test_process_method_rejected_under_ddp(
    yaak_dataset: Dataset, common_kwargs: dict[str, object]
) -> None:
    loader = TorchDataNodeDataLoader(
        dataset=yaak_dataset,
        method="process",
        multiprocessing_context="forkserver",
        **common_kwargs,  # ty:ignore[invalid-argument-type]
    )
    with pytest.raises(ValueError, match="deadlocks under distributed"):
        len(loader)
