from math import lcm
from typing import override

import polars as pl
from pydantic import PositiveInt, validate_call

from rbyte.io.table.base import Table

from .base import TableTransform


class FpsResampler(TableTransform):
    IDX_COL = "__idx"

    @validate_call
    def __init__(self, source_fps: PositiveInt, target_fps: PositiveInt) -> None:
        super().__init__()

        self._source_fps = source_fps
        self._target_fps = target_fps
        self._fps_lcm = lcm(source_fps, target_fps)

    @override
    def __call__(self, src: Table) -> Table:
        return (
            src.with_row_index(self.IDX_COL)
            .with_columns(pl.col(self.IDX_COL) * (self._fps_lcm // self._source_fps))
            .upsample(self.IDX_COL, every=f"{self._fps_lcm // self._target_fps}i")
            .interpolate()
            .fill_null(strategy="backward")
            .drop(self.IDX_COL)
            .drop_nulls()
        )
