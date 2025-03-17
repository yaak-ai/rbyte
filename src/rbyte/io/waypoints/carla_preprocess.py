import polars as pl


class CarlaWaypointsPreprocessor:
    """
    Preprocesses waypoints to be used in a model.
    """
    def __init__(self, n_wpts: int, idx_col: str, coords_col: str, heading_col: str,
                 ) -> None:
        self.n_wpts = n_wpts
        self._idx_col = idx_col
        self._coords_col = coords_col
        self._heading_col = heading_col

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._preprocess(input)

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        pass
