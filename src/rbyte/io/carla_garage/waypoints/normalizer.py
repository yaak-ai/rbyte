from typing import final
from uuid import uuid4

import polars as pl
from pydantic import BaseModel, validate_call
from structlog import get_logger

logger = get_logger(__name__)


class Columns(BaseModel):
    ego_coordinates: str
    waypoint_coordinates: str
    heading: str
    output: str


@final
class CarlaGarageWaypointNormalizer:
    __name__ = __qualname__

    INDEX_COLUMN = uuid4().hex

    @validate_call
    def __init__(self, columns: Columns) -> None:
        self._columns = columns

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._normalize(input)

    def _normalize(self, input: pl.DataFrame) -> pl.DataFrame:
        # TODO: simplify once geopolars (or equivalent) works  # noqa: FIX002

        match input.schema[self._columns.waypoint_coordinates]:
            case pl.Array(shape=(num_waypoints, 2)):
                pass

            case _:
                raise NotImplementedError

        x_waypoints, y_waypoints = (
            input[self._columns.waypoint_coordinates].explode().to_numpy().transpose()
        )

        y_ego, x_ego = (
            input.select(
                pl.col(self._columns.ego_coordinates)
                .arr.get(i)
                .repeat_by(num_waypoints)
                .explode()
            ).to_series()
            for i in range(2)
        )  # carla_garage has y/x coordinates pair

        df = (
            input.lazy()
            .select(
                pl.col(self._columns.heading)
                .repeat_by(num_waypoints)
                .flatten()
                .explode()
            )
            .with_columns(
                x_wpts=x_waypoints, y_wpts=y_waypoints, x_ego_pos=x_ego, y_ego_pos=y_ego
            )
            .with_columns(
                x_centered=pl.col("x_wpts") - pl.col("x_ego_pos"),
                y_centered=pl.col("y_wpts") - pl.col("y_ego_pos"),
            )
            .with_columns(
                x_centered_and_rotated=(
                    pl.col("x_centered") * pl.col(self._columns.heading).cos()
                    - pl.col("y_centered") * pl.col(self._columns.heading).sin()
                ),
                y_centered_and_rotated=(
                    pl.col("x_centered") * pl.col(self._columns.heading).sin()
                    + pl.col("y_centered") * pl.col(self._columns.heading).cos()
                ),
            )
            .with_columns(
                pl.concat_arr("x_centered_and_rotated", "y_centered_and_rotated").alias(
                    self._columns.output
                )
            )
            .with_row_index(name=self.INDEX_COLUMN)
            .cast({self.INDEX_COLUMN: pl.Int32})
            .group_by_dynamic(
                index_column=self.INDEX_COLUMN,
                period=f"{num_waypoints}i",
                every=f"{num_waypoints}i",
            )
            .agg(pl.col(self._columns.output))
            .drop(self.INDEX_COLUMN)
            .collect()
        )

        return input.drop(self._columns.waypoint_coordinates).hstack(df)
