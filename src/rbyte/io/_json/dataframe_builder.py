from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import final

import polars as pl
from optree import PyTree, tree_map
from polars._typing import PolarsDataType  # noqa: PLC2701
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from pydantic import ConfigDict, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars

from rbyte.utils.dataframe import unnest_all

logger = get_logger(__name__)


type Fields = Mapping[str, Mapping[str, PolarsDataType | None]]


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
        dfs: Mapping[str, pl.DataFrame] = {}

        for k, series in (
            pl.read_json(Path(path)).select(self._fields).to_dict().items()
        ):
            df_schema = {
                name: dtype
                for name, dtype in self._fields[k].items()
                if dtype is not None
            }
            df = pl.DataFrame(series).lazy().explode(k).unnest(k)
            dfs[k] = (
                df.select(unnest_all(df.collect_schema()))
                .select(self._fields[k].keys())
                .cast(df_schema)  # pyright: ignore[reportArgumentType]
                .collect()
            )

        return dfs  # pyright: ignore[reportReturnType]
