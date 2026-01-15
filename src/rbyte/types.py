from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar, runtime_checkable

from tensordict import NonTensorStack, TensorClass, TensorDict
from torch import Tensor


class BatchMeta(TensorClass, autocast=True):
    input_id: NonTensorStack


class Batch(TensorClass, autocast=True):
    data: TensorDict
    meta: BatchMeta | None = None


IndexT = TypeVar("IndexT")


@runtime_checkable
class TensorSource(Protocol, Generic[IndexT]):
    def __getitem__(self, indexes: IndexT | Sequence[IndexT]) -> Tensor: ...
    def __len__(self) -> int: ...
