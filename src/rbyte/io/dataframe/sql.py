from typing import final

import duckdb
import polars as pl
from pydantic import InstanceOf, validate_call


@final
class DataFrameDuckDbQuery:
    __name__ = __qualname__  # ty: ignore[unresolved-reference]

    @validate_call
    def __call__(
        self, *, query: str, **context: InstanceOf[pl.DataFrame]
    ) -> pl.DataFrame:
        with duckdb.connect() as con:
            for k, v in context.items():
                con.register(k, v)

            return con.sql(query).pl()
