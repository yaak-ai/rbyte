from typing import Annotated, override
from uuid import uuid4

import polars as pl
from pydantic import PositiveInt, StringConstraints, validate_call

from .base import SampleTableBuilder


class GreedySampleTableBuilder(SampleTableBuilder):
    @validate_call
    def __init__(
        self,
        index_column: str,
        length: PositiveInt = 1,
        min_step: PositiveInt = 1,
        stride: PositiveInt = 1,
        filter: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]  # noqa: A002
        | None = None,
    ) -> None:
        super().__init__()

        self._index_column = index_column
        self._length = length
        self._min_step = min_step
        self._stride = stride
        self._filter = filter

    @override
    def build(self, source: pl.LazyFrame) -> pl.LazyFrame:
        idx_col = self._index_column
        idx_dtype = source.select(idx_col).collect_schema()[idx_col]

        return (
            source.select(
                pl.int_range(
                    pl.col(idx_col).min().fill_null(value=0),
                    pl.col(idx_col).max().fill_null(value=0) + 1,
                    step=self._min_step,
                    dtype=idx_dtype,  # pyright: ignore[reportArgumentType]
                )
            )
            .select(
                pl.int_ranges(
                    pl.col(idx_col),
                    pl.col(idx_col) + self._length * self._stride,
                    self._stride,
                    dtype=idx_dtype,  # pyright: ignore[reportArgumentType]
                )
            )
            .with_row_index(sample_idx_col := uuid4().hex)
            .explode(idx_col)
            .join(source, on=idx_col, how="inner")
            .group_by(sample_idx_col)
            .all()
            .filter(pl.col(idx_col).list.len() == self._length)
            .sql(f"select * from self where ({self._filter or True})")  # noqa: S608
            .sort(sample_idx_col)
            .select(pl.exclude(sample_idx_col))
            # TODO: https://github.com/pola-rs/polars/issues/18810  # noqa: FIX002
            # .select(pl.all().list.to_array(self.length))
        )
