from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class TensorSource(Protocol):
    def __getitem__(self, indexes: Iterable[Any]) -> Tensor: ...
    def __len__(self) -> int: ...
