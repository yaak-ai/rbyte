import json
import math
import os
from collections.abc import Callable
from typing import Literal, final
from uuid import uuid4

import geopandas as gpd
import polars as pl
import shapely
from pyproj import Transformer
from structlog import get_logger

CoordinatesType = Literal["lat_lon", "xy"]


logger = get_logger(__name__)


@final
class DataFrameWaypointsMerger:
    """
    Merges waypoints from a separate file into a dataframe.
    """

    __name__ = __qualname__

    heading_col = "heading_rad"

    def __init__(
        self,
        waypoints_path: os.PathLike,
        coordinates_type: CoordinatesType,
        ego_lat_y_col: str,
        ego_lon_x_col: str,
        timestamp_col: str,
        num_waypoints: int = 10,
        out_col: str = "Waypoints.xy",
        predict_mode: bool = False,
        relative_to_wpts: bool = False,
    ) -> None:
        """
        We assume that wpts and ego coordinates are in the same CRS.
        """

        assert coordinates_type in ["lat_lon", "xy"], (
            "coordinates_type must be either 'lat_lon' or 'xy'"
        )

        match coordinates_type:
            # not necessary tho
            case "lat_lon":
                self.wpts_lon_x_col = "Waypoints.lon"
                self.wpts_lat_y_col = "Waypoints.lat"
            case "xy":
                self.wpts_lon_x_col = "Waypoints.x"
                self.wpts_lat_y_col = "Waypoints.y"

        self.ego_lon_x_col = ego_lon_x_col
        self.ego_lat_y_col = ego_lat_y_col

        self.wpts_mapview_col = f"Waypoints.mapview.{coordinates_type}"
        self.waypoints_path = waypoints_path
        self.timestamp_col = timestamp_col
        self.n_wpts = num_waypoints
        self.out_col = out_col
        self.predict_mode = predict_mode
        self.transform = self._transform_factory(coordinates_type)
        self.relative_to_wpts = relative_to_wpts

    @staticmethod
    def _transform_factory(coordinates_type: CoordinatesType) -> Transformer | Callable:
        # TODO: make it a separate function to call in pipeline
        match coordinates_type:
            case "lat_lon":
                return Transformer.from_crs(
                    "EPSG:4326", "EPSG:3857", always_xy=True
                ).transform
            case "xy":
                return lambda *args: args

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._merge_wpts(input)

    def _merge_wpts(self, input: pl.DataFrame) -> pl.DataFrame:
        df_wpts = self._build_df_wpts(self.waypoints_path)
        df = input.join_asof(
            df_wpts.select([
                self.timestamp_col,
                self.wpts_lat_y_col,
                self.wpts_lon_x_col,
                self.heading_col,
            ]),
            on=self.timestamp_col,
            strategy="nearest",
        )

        # TODO: get rid of this shit
        if self.relative_to_wpts:
            df = df.with_columns(
                pl.col("Waypoints.lat")
                .explode()
                .gather_every(10)
                .alias("Waypoints.lat.0"),
                pl.col("Waypoints.lon")
                .explode()
                .gather_every(10)
                .alias("Waypoints.lon.0"),
            )
            self.ego_lon_x_col = "Waypoints.lon.0"
            self.ego_lat_y_col = "Waypoints.lat.0"
        df = self._center_and_rotate(df=df, out_col=self.out_col)

        if self.predict_mode:
            # Some auxilary columns for rerun visualization
            logger.debug("Building auxilary columns")
            df = df.with_columns(
                pl.struct([self.wpts_lat_y_col, self.wpts_lon_x_col])
                .map_elements(
                    lambda row: list(
                        zip(
                            row[self.wpts_lat_y_col],
                            row[self.wpts_lon_x_col],
                            strict=False,
                        )
                    ),
                    return_dtype=pl.List(pl.List(pl.Float64)),
                )
                .alias(self.wpts_mapview_col)
            )

            df = self.build_heading_triangle(
                df,
                ego_lat_col=self.ego_lat_y_col,
                ego_lon_col=self.ego_lon_x_col,
                heading_col=self.heading_col,
            )
        logger.debug("waypoints merged")
        return df.drop(self.wpts_lat_y_col, self.wpts_lon_x_col)

    def _build_df_wpts(self, waypoints_path: os.PathLike) -> pl.DataFrame:
        df_wpts = self._json_to_df(waypoints_path)
        df_wpts = pl.concat(
            [df_wpts] + [df_wpts[-1, :]] * (self.n_wpts - 1)
        )  # duplicate last waypoints
        df_wpts = df_wpts.with_row_index().with_columns(
            index=pl.col("index").cast(pl.Int32)
        )
        for col in [self.wpts_lon_x_col, self.wpts_lat_y_col]:
            df_wpts = (
                df_wpts.group_by_dynamic(
                    index_column="index", period=f"{self.n_wpts}i", every="1i"
                )
                .agg(pl.col(col).alias(col))
                .join(df_wpts, on="index", how="left")
            )
        return df_wpts[: -(self.n_wpts - 1)]

    def _json_to_df(self, file_path: os.PathLike) -> pl.DataFrame:
        with open(file_path, "r") as file:
            geojson = json.load(file)

        data = {
            self.timestamp_col: [],
            self.heading_col: [],
            self.wpts_lon_x_col: [],
            self.wpts_lat_y_col: [],
        }
        for feature in geojson["features"]:
            data[self.timestamp_col].append(feature["properties"]["timestamp"])
            data[self.heading_col].append(
                math.radians(feature["properties"]["heading"])
            )
            data[self.wpts_lon_x_col].append(feature["geometry"]["coordinates"][0])
            data[self.wpts_lat_y_col].append(feature["geometry"]["coordinates"][1])
        df = pl.DataFrame(data)
        # WARN: Bad hack to handle idx type
        if self.timestamp_col != "_idx_":
            df = df.with_columns(
                pl.from_epoch(pl.col(self.timestamp_col) * 1e9, time_unit="ns").alias(
                    self.timestamp_col
                )
            )
        return df.sort(self.timestamp_col)

    def _center_and_rotate(self, df: pl.DataFrame, out_col: str) -> pl.DataFrame:
        """
        Usage of geopandas and shapely significantly improves performance.
        This function explodes lists of waypoints into one series of points and duplicate ego points accordingly.
        Then it calculates distance between each waypoint and ego position, rotates it and aggregate back into lists
        """

        # wpts
        x_wpts, y_wpts = self.transform(
            df[self.wpts_lon_x_col].explode(), df[self.wpts_lat_y_col].explode()
        )
        wpts_series = gpd.GeoSeries(shapely.points(x_wpts, y_wpts))

        # ego
        x_ego, y_ego = self.transform(
            df.select(
                pl.col(self.ego_lon_x_col).repeat_by(self.n_wpts).flatten().explode()
            ),
            df.select(
                pl.col(self.ego_lat_y_col).repeat_by(self.n_wpts).flatten().explode()
            ),
        )
        ego_pos_series = gpd.GeoSeries(shapely.points(x_ego, y_ego)[:, 0])

        x_diff = wpts_series.x - ego_pos_series.x
        y_diff = wpts_series.y - ego_pos_series.y
        headings = df.select(
            pl.col(self.heading_col).repeat_by(self.n_wpts).flatten().explode()
        )

        x_diff_col = uuid4().hex
        y_diff_col = uuid4().hex

        df_ = (
            pl.DataFrame({x_diff_col: x_diff, y_diff_col: y_diff, "_heading": headings})
            .with_columns(
                (
                    pl.col(x_diff_col) * pl.col("_heading").cos()
                    - pl.col(y_diff_col) * pl.col("_heading").sin()
                ).alias("x_diff"),
                (
                    pl.col(x_diff_col) * pl.col("_heading").sin()
                    + pl.col(y_diff_col) * pl.col("_heading").cos()
                ).alias("y_diff"),
            )
            .with_row_index()
            .with_columns(
                index=pl.col("index").cast(pl.Int32),
                xy=pl.concat_list([pl.col("x_diff"), pl.col("y_diff")]),
            )
            .drop([x_diff_col, y_diff_col, "_heading"])
            .group_by_dynamic(
                index_column="index", period=f"{self.n_wpts}i", every=f"{self.n_wpts}i"
            )
            .agg(pl.col("xy").list.to_array(width=2).alias(out_col))
            .drop("index")
        )
        return df.hstack(df_)

    @staticmethod
    def build_heading_triangle(
        df: pl.DataFrame,
        ego_lat_col: str,
        ego_lon_col: str,
        heading_col: str,
        l: float = 1.8 * 1e-4,  # approx 20 meters
    ) -> pl.DataFrame:
        """
        Build a triangle of points from the ego position and heading.
        """
        a_expr = [
            pl.col(ego_lat_col) + l * pl.col(heading_col).cos(),
            pl.col(ego_lon_col) + l * pl.col(heading_col).sin(),
        ]

        b_expr = [
            pl.col(ego_lat_col) - (l / 4) * pl.col(heading_col).sin(),
            pl.col(ego_lon_col) + (l / 4) * pl.col(heading_col).cos(),
        ]

        c_expr = [
            pl.col(ego_lat_col) + (l / 4) * pl.col(heading_col).sin(),
            pl.col(ego_lon_col) - (l / 4) * pl.col(heading_col).cos(),
        ]

        d_expr = [pl.col(ego_lat_col), pl.col(ego_lon_col)]

        return df.with_columns(
            pl.concat_list(
                a_expr[0],
                a_expr[1],
                b_expr[0],
                b_expr[1],
                c_expr[0],
                c_expr[1],
                d_expr[0],
                d_expr[1],
            ).alias("Waypoints.mapview.heading_triangle")
        )

    @staticmethod
    def _approximate_radius_deg(meters: float) -> float:
        lat = meters / 110574
        lon = meters / (111320 * math.cos(math.radians(lat)))
        return (lat + lon) / 2  # just a rough approximation since they are really close
