from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from tensordict import NonTensorStack, TensorClass, TensorDict
from torch import Tensor


class BatchMeta(TensorClass, autocast=True):
    input_id: NonTensorStack


class Batch(TensorClass, autocast=True):
    data: TensorDict
    meta: BatchMeta | None = None


@runtime_checkable
class TensorSource[I](Protocol):
    def __getitem__(self, indexes: I | Iterable[I]) -> Tensor: ...
    def __len__(self) -> int: ...
