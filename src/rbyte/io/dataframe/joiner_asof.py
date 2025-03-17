from collections.abc import Sequence
from datetime import timedelta
from typing import TypedDict, Unpack, final

import polars as pl
from polars import Expr
from polars._typing import AsofJoinStrategy, JoinValidation, MaintainOrderJoin
from pydantic import ConfigDict, validate_call
from structlog import get_logger

logger = get_logger(__name__)




class _Kwargs(TypedDict, total=False):
    on: str | Expr | Sequence[str | Expr] | None
    strategy: AsofJoinStrategy
    left_on: str | Expr | Sequence[str | Expr] | None
    right_on: str | Expr | Sequence[str | Expr] | None
    suffix: str
    validate: JoinValidation
    tolerance: str | int | float | timedelta | None
    nulls_equal: bool
    coalesce: bool | None
    maintain_order: MaintainOrderJoin | None


@final
class DataFrameJoinerAsof:
    __name__ = __qualname__

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, **kwargs: Unpack[_Kwargs]) -> None:
        self._kwargs = kwargs

    def __call__(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        out = left.join_asof(other=right, **self._kwargs)
        out.write_parquet('tmp/example.parquet')
        return out

