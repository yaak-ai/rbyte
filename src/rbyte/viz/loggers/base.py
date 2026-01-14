from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Logger[T](Protocol):
    def log(self, data: T) -> None: ...
