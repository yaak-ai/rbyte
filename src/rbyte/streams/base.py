from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class StreamSource[I](Protocol):
    def __getitem__(self, indexes: I | Sequence[I]) -> Tensor: ...
