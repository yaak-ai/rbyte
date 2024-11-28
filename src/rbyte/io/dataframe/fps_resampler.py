from math import lcm
from typing import final
from uuid import uuid4

import polars as pl
from pydantic import PositiveInt, validate_call


@final
class DataFrameFpsResampler:
    __name__ = __qualname__

    IDX_COL = uuid4().hex

    @validate_call
    def __init__(self, fps_in: PositiveInt, fps_out: PositiveInt) -> None:
        super().__init__()

        self._fps_in = fps_in
        self._fps_out = fps_out
        self._fps_lcm = lcm(fps_in, fps_out)

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return (
            input.with_row_index(self.IDX_COL)
            .with_columns(pl.col(self.IDX_COL) * (self._fps_lcm // self._fps_in))
            .upsample(self.IDX_COL, every=f"{self._fps_lcm // self._fps_out}i")
            .interpolate()
            .fill_null(strategy="backward")
            .drop(self.IDX_COL)
            .drop_nulls()
        )
