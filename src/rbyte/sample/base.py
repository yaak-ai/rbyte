from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class SampleTableBuilder(Protocol):
    def build(self, source: pl.LazyFrame) -> pl.LazyFrame: ...
