from collections.abc import Sequence
from typing import TypedDict, Unpack, final

import polars as pl
from polars import Expr
from polars._typing import JoinStrategy, JoinValidation, MaintainOrderJoin
from pydantic import ConfigDict, validate_call


class _Kwargs(TypedDict, total=False):
    on: str | Expr | Sequence[str | Expr] | None
    how: JoinStrategy
    left_on: str | Expr | Sequence[str | Expr] | None
    right_on: str | Expr | Sequence[str | Expr] | None
    suffix: str
    validate: JoinValidation
    nulls_equal: bool
    coalesce: bool | None
    maintain_order: MaintainOrderJoin | None


@final
class DataFrameJoiner:
    __name__ = __qualname__

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, **kwargs: Unpack[_Kwargs]) -> None:
        self._kwargs = kwargs

    def __call__(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        return left.join(other=right, **self._kwargs)
