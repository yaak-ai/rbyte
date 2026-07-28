from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from structlog import get_logger

from rbyte import Dataset
from rbyte.viz.loggers.rerun_logger import RerunLogger

logger = get_logger(__name__)

CONFIG_PATH = "../config"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _build_dataset(name: str) -> Dataset:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "dataset", overrides=[f"dataset={name}", f"+data_dir={DATA_DIR}/{name}"]
        )

    return instantiate(cfg.dataset)


# TODO: cleaner way of doing this while preserving fixture caching?  # ruff:ignore[line-contains-todo]
@pytest.fixture(scope="session")
def mimicgen_dataset() -> Dataset:
    return _build_dataset("mimicgen")


@pytest.fixture(scope="session")
def nuscenes_dataset() -> Dataset:
    return _build_dataset("nuscenes")


@pytest.fixture(scope="session")
def yaak_dataset() -> Dataset:
    return _build_dataset("yaak")


@pytest.fixture(scope="session")
def zod_dataset() -> Dataset:
    return _build_dataset("zod")


@pytest.fixture(params=["mimicgen", "nuscenes", "yaak", "zod"])
def rerun_logger(request: pytest.FixtureRequest) -> RerunLogger:
    name = request.param
    with initialize(version_base=None, config_path=f"{CONFIG_PATH}/logger/rerun"):
        cfg = compose(f"{name}", overrides=["++spawn=false"])

    return instantiate(cfg)
