from multiprocessing.context import ForkServerContext
from typing import Any, cast

import hydra
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from rbyte.viz.loggers.base import Logger

logger = get_logger(__name__)


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    logger = cast(Logger[Any], instantiate(config.logger))
    dataloader = cast(DataLoader[Any], instantiate(config.dataloader))

    if isinstance(dataloader.multiprocessing_context, ForkServerContext):  # pyright: ignore[reportUnknownMemberType]
        mp.set_forkserver_preload(["rbyte"])

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        logger.log(batch_idx, batch)


if __name__ == "__main__":
    main()
