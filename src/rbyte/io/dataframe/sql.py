from typing import final

import polars as pl
from pydantic import validate_call


@final
class DataFrameWithColumns:
    __name__ = __qualname__

    @validate_call
    def __init__(self, **sql_exprs: str) -> None:
        self._named_exprs = {k: pl.sql_expr(v) for k, v in sql_exprs.items()}

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return input.with_columns(**self._named_exprs)
