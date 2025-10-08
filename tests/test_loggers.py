import pytest
from pytest_lazy_fixtures import lf

from rbyte import Dataset
from rbyte.viz.loggers.rerun_logger import RerunLogger


@pytest.mark.parametrize(
    ("rerun_logger", "dataset"),
    [
        ("carla_garage", lf("carla_garage_dataset")),
        ("mimicgen", lf("mimicgen_dataset")),
        ("nuscenes", lf("nuscenes_dataset")),
        ("yaak", lf("yaak_dataset")),
        ("zod", lf("zod_dataset")),
    ],
    indirect=["rerun_logger"],
)
def test_rerun_logger(rerun_logger: RerunLogger, dataset: Dataset) -> None:
    rerun_logger.log(dataset.get_batch([0]))
