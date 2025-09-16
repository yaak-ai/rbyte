from math import ceil
from typing import Any, cast

import hydra
import more_itertools as mit
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from rbyte import Dataset
from rbyte.viz.loggers.base import Logger


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    logger = cast(Logger[Any], instantiate(config.logger))
    dataset = cast(Dataset, instantiate(config.dataset))

    dataset_len = len(dataset)
    batch_size = config.batch_size or dataset_len
    chunks = mit.chunked(range(dataset_len), batch_size)
    total = ceil(dataset_len / batch_size)

    for indexes in tqdm(chunks, total=total, disable=False):
        batch = dataset.get_batch(indexes)
        logger.log(batch)


if __name__ == "__main__":
    main()
