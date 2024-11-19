from datetime import timedelta
from typing import Literal, override
from uuid import uuid4

import polars as pl
from polars._typing import ClosedInterval
from pydantic import validate_call

from .base import SampleBuilder


class RollingWindowSampleBuilder(SampleBuilder):
    """
    Build samples using rolling windows based on a temporal or integer column.

    https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rolling
    """

    @validate_call
    def __init__(
        self,
        *,
        index_column: str,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        filter: str | None = None,  # noqa: A002
    ) -> None:
        self._index_column: pl.Expr = pl.col(index_column)
        self._period: str | timedelta = period
        self._offset: str | timedelta | None = offset
        self._closed: ClosedInterval = closed
        self._filter: str | Literal[True] = filter if filter is not None else True

    @override
    def build(self, source: pl.DataFrame) -> pl.DataFrame:
        return (
            source.sort(self._index_column)
            .with_columns(self._index_column.alias(_index_column := uuid4().hex))
            .rolling(
                index_column=_index_column,
                period=self._period,
                offset=self._offset,
                closed=self._closed,
            )
            .agg(pl.all())
            .sql(f"select * from self where ({self._filter})")  # noqa: S608
            .filter(self._index_column.list.len() > 0)
            .sort(_index_column)
            .drop(_index_column)
        )
