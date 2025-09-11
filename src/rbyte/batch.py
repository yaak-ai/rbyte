from typing import Literal

from tensordict import NonTensorData, TensorClass, TensorDict
from torch import Tensor


class BatchMeta(TensorClass, autocast=True):
    sample_idx: Tensor | None = None
    input_id: NonTensorData | None = None


class Batch(TensorClass, autocast=True):
    data: TensorDict | None = None
    meta: BatchMeta | None = None


type BatchKeys = frozenset[
    Literal["data", "meta"]
    | tuple[Literal["data"], str]
    | tuple[Literal["meta"], Literal["sample_idx", "input_id"]]
]

BATCH_KEYS_DEFAULT = frozenset(("data", "meta"))
