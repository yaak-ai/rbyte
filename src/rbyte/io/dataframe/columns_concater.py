from typing import final

from pydantic import validate_call

import polars as pl


@final
class DataFrameColumnsConcater:
    __name__ = __qualname__

    @validate_call
    def __init__(
        self,
        columns_in: list[str],
        col_out: str,
    ) -> None:
        self._columns_in = columns_in
        self._col_out = col_out

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return input.with_columns(
            pl.concat_arr(
                pl.col(col) for col in self._columns_in
            ).alias(self._col_out)
        )
