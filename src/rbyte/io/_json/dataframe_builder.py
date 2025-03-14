import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import ClassVar, final

import polars as pl
from optree import PyTree, PyTreeAccessor, tree_map, tree_map_with_accessor
from polars._typing import PolarsDataType  # noqa: PLC2701
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from pydantic import ConfigDict, RootModel, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars

logger = get_logger(__name__)


class Schema(RootModel[Mapping[str, PolarsDataType | None]]):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)


type Fields = Schema | Mapping[str, Fields]


@final
class JsonDataFrameBuilder:
    __name__ = __qualname__

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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
