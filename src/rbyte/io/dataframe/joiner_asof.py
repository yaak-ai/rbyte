from datetime import timedelta
from typing import TypedDict, Unpack, final

import polars as pl
from polars import Expr
from polars._typing import AsofJoinStrategy
from pydantic import ConfigDict, validate_call


class _Kwargs(TypedDict, total=False):
    on: str | Expr | None
    strategy: AsofJoinStrategy
    left_on: str | Expr | None
    right_on: str | Expr | None
    suffix: str
    tolerance: str | int | float | timedelta | None
    coalesce: bool


@final
class DataFrameJoinerAsof:
    __name__ = __qualname__

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, **kwargs: Unpack[_Kwargs]) -> None:
        self._kwargs = kwargs

    def __call__(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        return left.join_asof(other=right, **self._kwargs)
