from typing import final

from structlog import get_logger

import polars as pl

logger = get_logger(__name__)


@final
class WaypointsNormalizer:
    """
    Merges waypoints from a separate file into a dataframe.
    """

    __name__ = __qualname__
    idx_col = "_idx_"

    def __init__(
        self,
        ego_coords_col: str,
        waypoints_coords_col: str,
        heading_col: str,
        relative_to_wpts: bool = False,  # noqa: FBT001, FBT002
        out_col: str = "waypoints_normalized",
    ) -> None:
        self.ego_coords_col = ego_coords_col
        self.waypoints_coords_col = waypoints_coords_col
        self.heading_col = heading_col
        self.relative_to_wpts = relative_to_wpts
        self.out_col = out_col

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._center_and_rotate(input)

    def _center_and_rotate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Usage of geopandas and shapely significantly improves performance.
        This function explodes lists of waypoints into one series of points
          and duplicate ego points accordingly.
        Then it calculates distance between each waypoint and ego position,
          rotates it and aggregate back into lists.

        Returns:
            pl.DataFrame: A DataFrame with centered and rotated waypoints.
        """

        # get n_wpts
        wpts_list_len = df["waypoints.xy"].list.len()
        if wpts_list_len.is_duplicated().sum() != len(wpts_list_len):
            msg = "waypoints list length is not constant"
            logger.error(msg)
            raise ValueError(msg)
        n_wpts = wpts_list_len[0]

        wpts_exploded = df[self.waypoints_coords_col].explode()
        ego_pos_exploded = repeat_by_arr(df, self.ego_coords_col, n_wpts)
        headings_exploded = df.select(
            pl.col(self.heading_col).repeat_by(n_wpts).flatten().explode()
        ).to_series()

        x_centered = pl.Series(wpts_exploded.arr.get(0) - ego_pos_exploded.arr.get(0))
        y_centered = pl.Series(wpts_exploded.arr.get(1) - ego_pos_exploded.arr.get(1))

        x_centered_and_rotated = (
            x_centered * headings_exploded.cos() - y_centered * headings_exploded.sin()
        )
        y_centered_and_rotated = (
            x_centered * headings_exploded.sin() + y_centered * headings_exploded.cos()
        )

        df_ = (
            pl.DataFrame()
            .with_columns(
                pl.concat_list([x_centered_and_rotated, y_centered_and_rotated]).alias(
                    self.out_col
                )
            )
            .with_row_index(name=self.idx_col)
            .with_columns(pl.col(self.idx_col).cast(pl.Int32))
            .group_by_dynamic(
                index_column=self.idx_col, period=f"{n_wpts}i", every=f"{n_wpts}i"
            )
            .agg(pl.col(self.out_col).list.to_array(width=2))
            .drop(self.idx_col)
        )

        return df.hstack(df_)


def repeat_by_arr(df: pl.DataFrame, col: str, n: int) -> pl.Series:
    """
    `repeat_by` implementation for arrays
    TODO: replace with native polars `repeat_by` when 1.25.3 is released
    """

    df = df.with_columns(
        pl.col(col).explode().gather_every(2, offset=0).alias(f"{col}.x"),
        pl.col(col).explode().gather_every(2, offset=1).alias(f"{col}.y"),
    )
    df_exploded = df.select(
        pl.col(f"{col}.x").repeat_by(n).flatten(),
        pl.col(f"{col}.y").repeat_by(n).flatten(),
    )
    return df_exploded.with_columns(
        pl.concat_arr([pl.col(f"{col}.x"), pl.col(f"{col}.y")]).alias(col)
    )[col]
