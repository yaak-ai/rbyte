from typing import Protocol, runtime_checkable


@runtime_checkable
class Logger[T](Protocol):
    def log(self, batch_idx: int, batch: T) -> None: ...
