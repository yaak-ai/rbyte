from typing import Protocol, runtime_checkable

from rbyte.io.table.base import Table


@runtime_checkable
class TableTransform(Protocol):
    def __call__(self, src: Table) -> Table: ...
