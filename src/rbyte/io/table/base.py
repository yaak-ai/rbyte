from os import PathLike
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class TableBuilder(Protocol):
    def build(self, path: PathLike[str]) -> pl.DataFrame: ...
