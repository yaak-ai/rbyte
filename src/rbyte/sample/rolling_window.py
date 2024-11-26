from datetime import timedelta
from typing import final
from uuid import uuid4

import polars as pl
from polars._typing import ClosedInterval
from pydantic import validate_call


@final
class RollingWindowSampleBuilder:
    """
    Build samples using rolling windows based on a temporal or integer column.

    https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rolling
    """

    __name__ = __qualname__

    @validate_call
    def __init__(
        self,
        *,
        index_column: str,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
    ) -> None:
        self._index_column = pl.col(index_column)
        self._period = period
        self._offset = offset
        self._closed: ClosedInterval = closed

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return (
            input.sort(self._index_column)
            .with_columns(self._index_column.alias(_index_column := uuid4().hex))
            .rolling(
                index_column=_index_column,
                period=self._period,
                offset=self._offset,
                closed=self._closed,
            )
            .agg(pl.all())
            .filter(self._index_column.list.len() > 0)
            .sort(_index_column)
            .drop(_index_column)
        )
