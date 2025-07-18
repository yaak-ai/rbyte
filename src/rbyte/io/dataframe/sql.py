from collections.abc import Sequence
from typing import final

import duckdb
import polars as pl
from pydantic import ImportString, InstanceOf, validate_call

from rbyte.io._duckdb.udfs.base import DuckDbUdfKwargs


@final
class DataFrameDuckDbQuery:
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
    def __call__(
        self,
        *,
        query: str,
        df: InstanceOf[pl.DataFrame] | None = None,
        context: dict[str, InstanceOf[pl.DataFrame]] | None = None,
    ) -> pl.DataFrame:
        with duckdb.connect() as con:  # pyright: ignore[reportUnknownMemberType]
            match df, context:
                case [_, None]:
                    con.register("df", df)  # pyright: ignore[reportUnusedCallResult]

                case [None, _]:
                    for k, v in context.items():
                        con.register(k, v)  # pyright: ignore[reportUnusedCallResult]

                case _:
                    msg = "either `df` or `context` must be specified"
                    raise ValueError(msg)
            for udf in self.udfs:
                con.create_function(**udf)  # pyright: ignore[reportUnknownArgumentType, reportUnusedCallResult, reportArgumentType]
            return con.sql(query).pl()
