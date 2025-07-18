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
        self.udfs = udfs or []

    @validate_call
    def __call__(self, *, query: str, path: PathLike[str]) -> pl.DataFrame:
        with duckdb.connect() as con:  # pyright: ignore[reportUnknownMemberType]
            for udf in self.udfs:
                con.create_function(**udf)  # pyright: ignore[reportUnknownArgumentType, reportUnusedCallResult, reportArgumentType]

            return con.sql(query.format(path=path)).pl()
