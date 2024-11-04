from collections.abc import Hashable
from os import PathLike
from typing import Protocol, runtime_checkable

from optree import PyTree
from polars import DataFrame


@runtime_checkable
class TableBuilder(Protocol):
    def build(self) -> DataFrame: ...


@runtime_checkable
class TableReader(Protocol):
    def read(self, path: PathLike[str]) -> PyTree[DataFrame]: ...


@runtime_checkable
class TableMerger(Protocol):
    def merge(self, src: PyTree[DataFrame]) -> DataFrame: ...


@runtime_checkable
class TableCache(Protocol):
    def __contains__(self, key: Hashable) -> bool: ...
    def get(self, key: Hashable) -> DataFrame | None: ...
    def set(self, key: Hashable, value: DataFrame) -> bool: ...
