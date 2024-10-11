from collections.abc import Hashable, Mapping
from os import PathLike
from typing import Protocol, runtime_checkable

import polars as pl

Table = pl.DataFrame


@runtime_checkable
class TableBuilderBase(Protocol):
    def build(self) -> Table: ...


@runtime_checkable
class TableReaderBase(Hashable, Protocol):
    def read(self, path: PathLike[str]) -> Mapping[str, Table]: ...


@runtime_checkable
class TableMergerBase(Hashable, Protocol):
    def merge(self, src: Mapping[str, Table]) -> Table: ...


@runtime_checkable
class TableCacheBase(Protocol):
    def __contains__(self, key: Hashable) -> bool: ...
    def get(self, key: Hashable) -> Table | None: ...
    def set(self, key: Hashable, value: Table) -> bool: ...
