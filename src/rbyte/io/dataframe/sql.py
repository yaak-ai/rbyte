from collections.abc import Sequence
from typing import final

import duckdb
import polars as pl
from pydantic import ImportString, InstanceOf, validate_call

from rbyte.io._duckdb.udfs.base import DuckDbUdfKwargs, register_duckdb_udf


@final
class DataFrameDuckDbQuery:
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
    def __call__(
        self,
        *,
        query: str,
        df: InstanceOf[pl.DataFrame] | None = None,
        context: dict[str, InstanceOf[pl.DataFrame]] | None = None,
    ) -> pl.DataFrame:
        match df, context:
            case [_, None]:
                duckdb.register("df", df)  # pyright: ignore[reportUnusedCallResult]

            case [None, _]:
                for k, v in context.items():
                    duckdb.register(k, v)  # pyright: ignore[reportUnusedCallResult]

            case _:
                msg = "either `df` or `context` must be specified"
                raise ValueError(msg)

        return duckdb.sql(query).pl()
