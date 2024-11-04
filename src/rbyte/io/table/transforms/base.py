from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class TableTransform(Protocol):
    def __call__(self, src: pl.DataFrame) -> pl.DataFrame: ...
