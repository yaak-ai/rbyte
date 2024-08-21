from collections.abc import Callable
from typing import cast

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from polars import DataFrame
from structlog import get_logger

from rbyte.io.table.base import TableBuilder

logger = get_logger(__name__)


def run(config: DictConfig) -> None:
    builder = cast(TableBuilder, instantiate(config.builder))
    writer = cast(Callable[[DataFrame], None], instantiate(config.writer))
    df = builder.build(config.path)

    return writer(df)


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    try:
        run(config)
    except Exception:
        logger.exception("failed")


if __name__ == "__main__":
    main()
