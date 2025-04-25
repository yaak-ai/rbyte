from collections.abc import Mapping
from typing import final

import duckdb
import polars as pl
from pydantic import InstanceOf, validate_call


@final
class DataFrameDuckDbQuery:
    __name__ = __qualname__

    @validate_call
    def __call__(
        self, *, query: str, context: Mapping[str, InstanceOf[pl.DataFrame]]
    ) -> pl.DataFrame:
        for k, v in context.items():
            duckdb.register(k, v)  # pyright: ignore[reportUnusedCallResult]

        return duckdb.sql(query).pl()
