from typing import Any, cast

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from rbyte.viz.loggers.base import Logger


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    logger = cast(Logger[Any], instantiate(config.logger))
    dataloader = cast(DataLoader[Any], instantiate(config.dataloader))

    for batch in tqdm(dataloader, disable=False):
        logger.log(batch)


if __name__ == "__main__":
    main()
