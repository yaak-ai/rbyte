from typing import Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Logger(Protocol, Generic[T]):
    def log(self, data: T) -> None: ...
