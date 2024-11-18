from typing import Annotated, override
from uuid import uuid4

import polars as pl
from pydantic import PositiveInt, StringConstraints, validate_call

from .base import SampleBuilder


class GreedySampleBuilder(SampleBuilder):
    @validate_call
    def __init__(
        self,
        index_column: str | None = None,
        length: PositiveInt = 1,
        min_step: PositiveInt = 1,
        stride: PositiveInt = 1,
        filter: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]  # noqa: A002
        | None = None,
    ) -> None:
        super().__init__()

        self._index_column: str | None = index_column
        self._length: int = length
        self._min_step: int = min_step
        self._stride: int = stride
        self._filter: str | None = filter

    @override
    def build(self, source: pl.LazyFrame) -> pl.LazyFrame:
        if (idx_col := self._index_column) is None:
            idx_col = uuid4().hex
            source = source.with_row_index(idx_col)

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
            .drop([sample_idx_col, *([idx_col] if self._index_column is None else [])])
            .select(pl.all().list.to_array(self._length))
        )
