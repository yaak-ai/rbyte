import pytest
from pytest_lazy_fixtures import lf
from torch.utils.data import DataLoader

from rbyte import Dataset
from rbyte.dataloader import NodeDataLoader, collate_identity


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
) -> NodeDataLoader:
    match request.param:
        case "process":
            kwargs = {"multiprocessing_context": "forkserver", "method": "process"}

        case "thread":
            kwargs = {"method": "thread"}

        case _:
            raise RuntimeError

    return NodeDataLoader(
        dataset=dataset,
        **(common_kwargs | kwargs),  # ty:ignore[invalid-argument-type]
    )


@pytest.mark.parametrize("dataset", [lf("yaak_dataset")])
def test_dataloaders(
    torch_dataloader: DataLoader, torchdata_dataloader: NodeDataLoader
) -> None:
    for left, right in zip(torch_dataloader, torchdata_dataloader, strict=True):
        assert (left == right).all()
