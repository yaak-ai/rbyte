from collections.abc import Sequence
from os import PathLike
from typing import final

import duckdb
import polars as pl
from pydantic import ImportString, validate_call

from .udfs.base import DuckDbUdfKwargs


@final
class DuckDbDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(
        self,
        *,
        udfs: Sequence[ImportString[DuckDbUdfKwargs]]
        | Sequence[DuckDbUdfKwargs]
        | None = None,
    ) -> None:
        for udf in udfs or []:
            try:
                _ = duckdb.create_function(**udf)  # pyright: ignore[reportArgumentType]
            except duckdb.NotImplementedException:
                _ = duckdb.remove_function(udf["name"])
                _ = duckdb.create_function(**udf)  # pyright: ignore[reportArgumentType]

    @validate_call
    def __call__(self, *, query: str, path: PathLike[str]) -> pl.DataFrame:
        return duckdb.sql(query=query.format(path=path)).pl()
