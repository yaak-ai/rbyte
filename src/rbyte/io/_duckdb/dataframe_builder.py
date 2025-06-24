from collections.abc import Sequence
from os import PathLike
from typing import final

import duckdb
import polars as pl
from pydantic import ImportString, validate_call

from .udfs.base import DuckDbUdfKwargs, register_duckdb_udf


@final
class DuckDbDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(
        self,
        *,
        udfs: Sequence[ImportString[type[DuckDbUdfKwargs]] | DuckDbUdfKwargs]
        | None = None,
    ) -> None:
        register_duckdb_udf(udfs)

    @validate_call
    def __call__(self, *, query: str, path: PathLike[str]) -> pl.DataFrame:
        return duckdb.sql(query=query.format(path=path)).pl()
