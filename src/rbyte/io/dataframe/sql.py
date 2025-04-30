from typing import final

import duckdb
import polars as pl
from pydantic import InstanceOf, validate_call


@final
class DataFrameDuckDbQuery:
    __name__ = __qualname__

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
