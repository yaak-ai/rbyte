from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class SampleBuilder(Protocol):
    def build(self, source: pl.DataFrame) -> pl.DataFrame: ...
