from functools import partial
from typing import TYPE_CHECKING, final

import polars as pl
from pydantic import validate_call

if TYPE_CHECKING:
    from collections.abc import Callable


class DataFrameQuery:
    __name__: str = __qualname__

    @validate_call
    def __init__(self, *, query: str, table_name: str = "self") -> None:
        self._fn: Callable[[pl.DataFrame], pl.DataFrame] = partial(
            pl.DataFrame.sql, query=query, table_name=table_name
        )

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._fn(input)


@final
class DataFrameFilter(DataFrameQuery):
    __name__ = __qualname__

    @validate_call
    def __init__(self, predicate: str) -> None:
        super().__init__(
            query=f"select * from {(table_name := 'self')} where {predicate}",  # noqa: S608
            table_name=table_name,
        )


@final
class DataFrameWithColumns:
    __name__ = __qualname__

    @validate_call
    def __init__(self, **sql_exprs: str) -> None:
        self._named_exprs = {k: pl.sql_expr(v) for k, v in sql_exprs.items()}

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return input.with_columns(**self._named_exprs)
