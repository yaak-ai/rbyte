from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class TensorSource[I](Protocol):
    def __getitem__(self, indexes: I | Sequence[I]) -> Tensor: ...
    def __len__(self) -> int: ...
