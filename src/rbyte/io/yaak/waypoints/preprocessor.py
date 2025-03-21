from typing import final
from uuid import uuid4

import polars as pl
from pydantic import validate_call

from rbyte.config.base import BaseModel


class Columns(BaseModel):
    timestamp: str
    heading: str
    coordinates: str
    output: str


@final
class YaakWaypointPreprocessor:
    __name__ = __qualname__

    INDEX_COLUMN = uuid4().hex

    @validate_call
    def __init__(self, num_waypoints: int, columns: Columns) -> None:
        self._num_waypoints = num_waypoints
        self._columns = columns

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        df = self._preprocess(input)
        return self._sample_waypoints(df)

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.lazy()
            .with_columns(
                pl.col(self._columns.heading).radians(),
                pl.col(self._columns.timestamp)
                .pipe(pl.from_epoch, time_unit="s")
                .cast(pl.Datetime("ns")),
            )
            .sort(self._columns.timestamp)
            .collect()
        )

    def _sample_waypoints(self, df: pl.DataFrame) -> pl.DataFrame:
        df_ = (
            pl.concat([df] + [df[-1, :]] * (self._num_waypoints - 1))
            .lazy()
            .with_row_index(name=self.INDEX_COLUMN)
            .cast({self.INDEX_COLUMN: pl.Int32})
        )

        return (
            df_.group_by_dynamic(
                index_column=self.INDEX_COLUMN,
                period=f"{self._num_waypoints}i",
                every="1i",
            )
            .agg(pl.col(self._columns.coordinates).alias(self._columns.output))
            .join(df_, on=self.INDEX_COLUMN, how="left")
            .drop(self.INDEX_COLUMN)
            .collect()
            .slice(0, -(self._num_waypoints - 1))
            .with_columns(
                pl.col(self._columns.output).list.to_array(width=self._num_waypoints)
            )
        )
