from typing import Annotated, override
from uuid import uuid4

import polars as pl
from pydantic import PositiveInt, StringConstraints, validate_call

from rbyte.config.base import BaseModel

from .base import SampleTableBuilder


class GreedySampleTableBuilderConfig(BaseModel):
    index_column: str
    length: PositiveInt
    min_step: PositiveInt
    stride: PositiveInt = 1
    filter: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | None
    ) = None


class GreedySampleTableBuilder(SampleTableBuilder):
    @validate_call
    def __init__(self, config: object) -> None:
        super().__init__()

        self._config = GreedySampleTableBuilderConfig.model_validate(config)

    @property
    def config(self) -> GreedySampleTableBuilderConfig:
        return self._config

    @override
    def build(self, source: pl.LazyFrame) -> pl.LazyFrame:
        idx_col = self.config.index_column
        idx_dtype = source.select(idx_col).collect_schema()[idx_col]

        return (
            source.select(
                pl.int_range(
                    pl.col(idx_col).min().fill_null(value=0),
                    pl.col(idx_col).max().fill_null(value=0) + 1,
                    step=self.config.min_step,
                    dtype=idx_dtype,  # pyright: ignore[reportArgumentType]
                )
            )
            .select(
                pl.int_ranges(
                    pl.col(idx_col),
                    pl.col(idx_col) + self.config.length * self.config.stride,
                    self.config.stride,
                    dtype=idx_dtype,  # pyright: ignore[reportArgumentType]
                )
            )
            .with_row_index(sample_idx_col := uuid4().hex)
            .explode(idx_col)
            .join(source, on=idx_col, how="inner")
            .group_by(sample_idx_col)
            .all()
            .filter(pl.col(idx_col).list.len() == self.config.length)
            .sql(f"select * from self where ({self.config.filter or True})")  # noqa: S608
            .sort(sample_idx_col)
            .select(pl.exclude(sample_idx_col).list.to_array(self.config.length))
        )
