import polars as pl


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
        idx_col: str = "_idx_",
        datetime_col: str = "datetime",
        waypoints_col: str = "waypoints",
    ) -> None:
        self.n_wpts = n_wpts
        self._idx_col = idx_col
        self._heading_col = heading_col
        self._timestamp_col = timestamp_col
        self._coords_col = coords_col
        self._datetime_col = datetime_col
        self._waypoints_col = waypoints_col

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._preprocess(input)

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Preprocesses the input DataFrame by performing the following steps:
        
        1. Converts the heading column values from degrees to radians.
        2. Converts the timestamp column values from epoch time to datetime.
        3. Sorts the DataFrame based on the datetime column.
        4. Duplicates the last waypoints to ensure there are enough waypoints for processing.
        5. Adds a row index column and casts it to Int32.
        6. Groups the DataFrame dynamically based on the row index column.
        7. Aggregates the coordinates column within each group.
        8. Joins the aggregated DataFrame with the original DataFrame on the row index column.
        9. Removes the duplicated last waypoints to return the DataFrame to its original length.
        
        Args:
            df (pl.DataFrame): The input DataFrame containing waypoints data.
        
        Returns:
            pl.DataFrame: The preprocessed DataFrame with the necessary transformations applied.
        """
        df = df.with_columns(
            pl.col(self._heading_col).radians(),
            pl.col(self._timestamp_col)
            .pipe(pl.from_epoch, time_unit="s")
            .cast(pl.Datetime("ns"))
            .alias(self._datetime_col),
        ).sort(self._datetime_col)

        # QUESTION: why do we need to duplicate last waypoints and then delete them?
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
        )
        return df[: -(self.n_wpts - 1)]
