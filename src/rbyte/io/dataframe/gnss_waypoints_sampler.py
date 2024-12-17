from typing import final

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


# TODO:
# - [ ] add pydantic config


@final
class DataFrameGnssWaypointsSampler:
    __name__ = __qualname__

    def __init__(
        self,
        columns: dict[str, str],
        num_waypoints: int = 20,
        time_window_seconds: int = 120,
        radius_meters: int = 200,
    ) -> None:
        self.columns: dict[str, str] = columns
        self.num_waypoints: int = num_waypoints
        self.time_window_seconds: int = time_window_seconds
        self.radius_degrees: float = self._approximate_radius_deg(radius_meters)

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        df_input = (
            input.select(self.columns.values())
            .rename({v: k for k, v in self.columns.items()})
            .lazy()
        )
        # generate rolling window
        df = (
            df_input.rolling(
                "time_stamp",
                offset=f"-{int(self.time_window_seconds / 2)}s",
                period=f"{self.time_window_seconds}s",
            )
            .agg([pl.struct(["latitude", "longitude"]).alias("rolling_lat_lon")])
            .explode("rolling_lat_lon")
            .unnest("rolling_lat_lon")
            .rename({"latitude": "rolling_lat", "longitude": "rolling_lon"})
            .join(df_input, on="time_stamp", how="left")
        )

        # filter by radius
        df = (
            df.filter(
                (pl.col("rolling_lat") - pl.col("latitude")) ** 2
                + (pl.col("rolling_lon") - pl.col("longitude")) ** 2
                < self.radius_degrees**2
            )
            .select(["time_stamp", "rolling_lat", "rolling_lon"])
            .group_by("time_stamp")
            .agg([
                pl.col("rolling_lat").alias("rolling_lat"),
                pl.col("rolling_lon").alias("rolling_lon"),
            ])
            .join(df_input.lazy(), on="time_stamp", how="left")
        )

        # sample equidistant points
        # it takes the longest. rewrite
        df = (
            df.with_columns(
                pl.struct(["rolling_lon", "rolling_lat"])
                .map_elements(
                    self.sample_equidistant_points, return_dtype=pl.List(pl.Float64)
                )
                .alias("sampled_points")
            )
            .with_columns([
                pl.col("sampled_points")
                .list.slice(0, self.num_waypoints)
                .alias("waypoints_longitude"),
                pl.col("sampled_points")
                .list.slice(self.num_waypoints, self.num_waypoints)
                .alias("waypoints_latitude"),
            ])
            .drop("sampled_points", "rolling_lat", "rolling_lon")
        ).collect()
        self._check_df(df, num_waypoints=self.num_waypoints)
        logger.debug("waypoints sampled")
        return (
            df.select(pl.col("time_stamp", "waypoints_longitude", "waypoints_latitude"))
            .rename({"time_stamp": self.columns["time_stamp"]})
            .join(input, on=self.columns["time_stamp"], how="left")
        )

    def sample_equidistant_points(self, row: list[list[float]]) -> np.ndarray:
        """
        Sample a fixed number of equidistant points from a set of (x, y) coordinates.

        Parameters:
        points (np.ndarray): An array of shape (n, 2) containing (x, y) coordinates.
        num_samples (int): The desired number of equidistant points.

        Returns:
        np.ndarray: An array of shape (num_samples, 2) containing the sampled equidistant points.
        """
        points: np.ndarray[float] = np.column_stack((
            row["rolling_lon"],
            row["rolling_lat"],
        ))

        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        total_length = cumulative_distances[-1]
        sample_distances: np.ndarray[float] = np.linspace(
            0, total_length, self.num_waypoints
        )

        return np.column_stack((
            np.interp(sample_distances, cumulative_distances, points[:, 0]),
            np.interp(sample_distances, cumulative_distances, points[:, 1]),
        )).flatten(order="F")

    @staticmethod
    def _approximate_radius_deg(meters: float) -> float:
        lat = meters / 110574
        lon = meters / (111320 * np.cos(np.radians(lat)))
        return (lat + lon) / 2  # just a rough approximation since they are really close

    @staticmethod
    def _check_df(df: pl.DataFrame, num_waypoints: int) -> None:
        if df["waypoints_longitude"].is_null().sum() > 0:
            logger.error(
                msg := "There are null values in the 'waypoints_longitude' column."
            )
            raise ValueError(msg)

        if df["waypoints_latitude"].is_null().sum() > 0:
            logger.error(
                msg := "There are null values in the 'waypoints_latitude' column."
            )
            raise ValueError(msg)

        if df["time_stamp"].unique().len() != df.shape[0]:
            logger.error(
                msg
                := "The number of unique timestamps in df_final does not match the number of rows."
            )
            raise ValueError(msg)

        if not df["time_stamp"].is_sorted():
            logger.error(msg := "The 'time_stamp' column in df_final is not sorted.")
            raise ValueError(msg)

        if not (
            df["waypoints_longitude"].list.len().min()
            == df["waypoints_longitude"].list.len().max()
            == num_waypoints
        ):
            logger.error(
                msg
                := "The lengths of lists in 'waypoints_longitude' column are not equal to num_waypoints."
            )
            raise ValueError(msg)

        if not (
            df["waypoints_latitude"].list.len().min()
            == df["waypoints_latitude"].list.len().max()
            == num_waypoints
        ):
            logger.error(
                msg
                := "The lengths of lists in 'waypoints_latitude' column are not equal to num_waypoints."
            )
            raise ValueError(msg)
