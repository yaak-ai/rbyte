import json
from os import PathLike
from pathlib import Path
from typing import final

import polars as pl
from optree import PyTree, PyTreeAccessor, tree_map, tree_map_with_accessor
from polars.datatypes import DataType
from pydantic import InstanceOf, RootModel, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars

logger = get_logger(__name__)


class Schema(RootModel[dict[str, InstanceOf[DataType] | None]]): ...


type Fields = Schema | dict[str, Fields]


@final
class JsonDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(self, fields: Fields) -> None:
        self._fields = fields

    def __call__(self, path: PathLike[str]) -> PyTree[pl.DataFrame]:
        with bound_contextvars(path=path):
            result = self._build(path)
            logger.debug("built dataframes", length=tree_map(len, result))

            return result

    def _build(self, path: PathLike[str]) -> PyTree[pl.DataFrame]:
        with Path(path).open(encoding=None) as fp:
            data = json.load(fp)

        def json_normalize(accessor: PyTreeAccessor, schema: Schema) -> pl.DataFrame:
            return pl.json_normalize(accessor(data), schema=schema.root)  # pyright: ignore[reportArgumentType]

        return tree_map_with_accessor(
            json_normalize,
            self._fields,  # pyright: ignore[reportArgumentType]
            is_leaf=lambda x: isinstance(x, Schema),
        )
