from typing import final

import polars as pl


@final
class YaakWaypointsPreprocessor:
    """
    Preprocesses waypoints to be used in a model.
    """

    __name__ = __qualname__

    def __init__(
        self,
        n_wpts: int,
        heading_col: str,
        timestamp_col: str,
        coords_col: str,
        waypoints_col: str = "waypoints",
    ) -> None:
        self.n_wpts = n_wpts
        self._heading_col = heading_col
        self._timestamp_col = timestamp_col
        self._coords_col = coords_col
        self._waypoints_col = waypoints_col
        self._idx_col = "_idx_"

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._preprocess(input)

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(self._heading_col).radians(),
            pl.col(self._timestamp_col)
            .pipe(pl.from_epoch, time_unit="s")
            .cast(pl.Datetime("ns")),
        ).sort(self._timestamp_col)

        df = (
            pl.concat(
                [df] + [df[-1, :]] * (self.n_wpts - 1)
            )  # duplicate last waypoints
            .with_row_index(name=self._idx_col)
            .with_columns(
                pl.col(self._idx_col).cast(pl.Int32)
            )  # cast to int32, otherwise it will be uint32
        )
        df = (
            df.group_by_dynamic(
                index_column=self._idx_col, period=f"{self.n_wpts}i", every="1i"
            )
            .agg(pl.col(self._coords_col).alias(self._waypoints_col))
            .join(df, on=self._idx_col, how="left")
            .drop(self._idx_col)
        )
        return df[: -(self.n_wpts - 1)]
