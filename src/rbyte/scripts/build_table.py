from collections.abc import Callable
from typing import cast

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger

from rbyte.io.table.base import Table, TableBuilderBase

logger = get_logger(__name__)


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    table_builder = cast(TableBuilderBase, instantiate(config.table_builder))
    table_writer = cast(Callable[[Table], None], instantiate(config.table_writer))
    table = table_builder.build()

    return table_writer(table)


if __name__ == "__main__":
    main()
