from datetime import timedelta
from typing import final
from uuid import uuid4

import polars as pl
from polars._typing import ClosedInterval
from pydantic import PositiveInt, validate_call


@final
class FixedWindowSampleBuilder:
    """
    Build samples using fixed (potentially overlapping) windows based on a temporal or
    integer column.

    https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by_dynamic
    """

    __name__ = __qualname__

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        index_column: str,
        every: str | timedelta,
        period: str | timedelta | None = None,
        closed: ClosedInterval = "left",
        gather_every: PositiveInt = 1,
        length: PositiveInt | None = None,
    ) -> None:
        self._index_column = pl.col(index_column)
        self._every = every
        self._period = period
        self._closed: ClosedInterval = closed
        self._gather_every = gather_every
        self._length_filter = (
            (self._index_column.list.len() > 0)
            if length is None
            else (self._index_column.list.len() == length)
        )

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return (
            input.sort(self._index_column)
            .with_columns(self._index_column.alias(_index_column := uuid4().hex))
            .group_by_dynamic(
                index_column=_index_column,
                every=self._every,
                period=self._period,
                closed=self._closed,
                label="datapoint",
                start_by="datapoint",
            )
            .agg(pl.all().gather_every(self._gather_every))
            .filter(self._length_filter)
            .drop(_index_column)
        )
