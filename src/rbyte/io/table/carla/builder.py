from collections.abc import Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Annotated, override

import polars as pl
from polars import selectors as cs
from pydantic import StringConstraints

from rbyte.config.base import BaseModel, HydraConfig
from rbyte.io.table.base import TableBuilder
from rbyte.io.table.transforms.base import TableTransform
from rbyte.utils.dataframe import unnest_all


class CarlaRecordsTableBuilderConfig(BaseModel):
    index_column: Annotated[str, StringConstraints(strip_whitespace=True)] = "frame_idx"
    select: str | frozenset[str] = "*"
    filter: Annotated[str, StringConstraints(strip_whitespace=True)] | None = None
    transforms: tuple[HydraConfig[TableTransform], ...] = ()


class CarlaRecordsTableBuilder(TableBuilder):
    RECORD_KEY = "records"

    def __init__(self, config: object) -> None:
        super().__init__()

        self._config = CarlaRecordsTableBuilderConfig.model_validate(config)

    @property
    def config(self) -> CarlaRecordsTableBuilderConfig:
        return self._config

    @cached_property
    def _transforms(self) -> Sequence[TableTransform]:
        return tuple(transform.instantiate() for transform in self.config.transforms)

    @override
    def build(self, path: PathLike[str]) -> pl.DataFrame:
        df = pl.read_json(Path(path)).explode(self.RECORD_KEY).unnest(self.RECORD_KEY)
        df = (
            df.select(unnest_all(df.collect_schema()))
            .select(self.config.select)
            # 32 bits ought to be enough for anybody
            .cast({
                cs.by_dtype(pl.Int64): pl.Int32,
                cs.by_dtype(pl.UInt64): pl.UInt32,
                cs.by_dtype(pl.Float64): pl.Float32,
            })
            .sql(f"select * from self where ({self.config.filter or True})")  # noqa: S608
        )

        for transform in self._transforms:
            df = transform(df)

        return df.with_row_index(self.config.index_column)
