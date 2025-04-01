from collections.abc import Sequence
from typing import TypedDict, Unpack, final

import polars as pl
from polars._typing import (
    JoinStrategy,  # pyright: ignore[reportPrivateImportUsage]
    JoinValidation,  # pyright: ignore[reportPrivateImportUsage]
    MaintainOrderJoin,  # pyright: ignore[reportPrivateImportUsage]
)
from pydantic import validate_call


class _Kwargs(TypedDict, total=False):
    on: str | Sequence[str] | None
    how: JoinStrategy
    left_on: str | Sequence[str] | None
    right_on: str | Sequence[str] | None
    suffix: str
    validate: JoinValidation
    nulls_equal: bool
    coalesce: bool | None
    maintain_order: MaintainOrderJoin | None


@final
class DataFrameJoiner:
    __name__ = __qualname__

    @validate_call
    def __init__(self, **kwargs: Unpack[_Kwargs]) -> None:
        self._kwargs = kwargs

    def __call__(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        return left.join(other=right, **self._kwargs)
