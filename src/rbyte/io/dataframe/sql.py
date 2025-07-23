from typing import final

import duckdb
import polars as pl
from pydantic import InstanceOf, validate_call


@final
class DataFrameDuckDbQuery:
    __name__ = __qualname__

    @validate_call
    def __call__(
        self, *, query: str, **context: InstanceOf[pl.DataFrame]
    ) -> pl.DataFrame:
        with duckdb.connect() as con:  # pyright: ignore[reportUnknownMemberType]
            for k, v in context.items():
                con.register(k, v)  # pyright: ignore[reportUnusedCallResult]

            return con.sql(query).pl()
