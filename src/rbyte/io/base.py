from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class TensorSource(Protocol):
    def __getitem__[T](self, indexes: T | Sequence[T]) -> Tensor: ...
    def __len__(self) -> int: ...
