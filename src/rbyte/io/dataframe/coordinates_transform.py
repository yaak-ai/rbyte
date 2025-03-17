from typing import Literal, final

import polars as pl
from pyproj import Transformer
from structlog import get_logger

logger = get_logger(__name__)


@final
class DataFrameCoordinatesTransform:
    __name__ = __qualname__
    """
    Wrapper around pyproj.Transformer
    Transforms lat/lon coordinates to x/y coordinates.
    """

    def __init__(
        self,
        col_coords: str,
        format_coords: Literal["lon_lat", "lat_lon"],
        col_out_xy: str = "xy",
        crs_from: str = "EPSG:4326",
        crs_to: str = "EPSG:3857",
    ) -> None:
        self.col_coords = col_coords
        self.col_out_xy = col_out_xy

        if format_coords == "lon_lat":
            self.lon_pos = 0
            self.lat_pos = 1
        else:
            self.lon_pos = 1
            self.lat_pos = 0
        self._transform = Transformer.from_crs(
            crs_from=crs_from, crs_to=crs_to, always_xy=True
        ).transform

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._transform_df(input)

    def _transform_df(self, input: pl.DataFrame) -> pl.DataFrame:
        x, y = self._transform(
            input[self.col_coords].arr.get(self.lon_pos),
            input[self.col_coords].arr.get(self.lat_pos),
        )

        return input.with_columns(
            pl.concat_arr(pl.Series(values=x), pl.Series(values=y)).alias(
                self.col_out_xy
            )
        )
