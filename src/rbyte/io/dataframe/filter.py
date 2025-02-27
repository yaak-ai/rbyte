from typing import final

import polars as pl


@final
class DataFrameFilter:
    __name__ = __qualname__

    def __init__(self, predicate: str) -> None:
        self._query = f"select * from self where {predicate}"  # noqa: S608

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return input.sql(self._query)
