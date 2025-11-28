import pytest
from pytest_lazy_fixtures import lf
from torch.utils.data import DataLoader

from rbyte import Dataset
from rbyte.dataloader import TorchDataNodeDataLoader, collate_identity


@pytest.mark.parametrize("dataset", [lf("yaak_dataset")])
def test_dataloaders(dataset: Dataset) -> None:
    kwargs = {
        "dataset": dataset,
        "batch_size": 2,
        "shuffle": False,
        "collate_fn": collate_identity,
        "num_workers": 1,
        "multiprocessing_context": "forkserver",
    }

    torch_dataloader = DataLoader(**kwargs)  # ty: ignore[invalid-argument-type]
    torchdata_dataloader = TorchDataNodeDataLoader(method="process", **kwargs)  # ty: ignore[invalid-argument-type]

    for left, right in zip(torch_dataloader, torchdata_dataloader, strict=True):
        assert (left == right).all()
