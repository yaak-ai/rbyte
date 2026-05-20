from pathlib import Path
from typing import ClassVar

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from structlog import get_logger

from rbyte import Dataset
from rbyte.config import HydraConfig

logger = get_logger(__name__)


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    dataset: HydraConfig[Dataset]
    path: Path


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # ty:ignore[invalid-argument-type]
    config.path.mkdir(parents=True, exist_ok=True)
    dataset = config.dataset.instantiate()
    dataset.save(config.path)
