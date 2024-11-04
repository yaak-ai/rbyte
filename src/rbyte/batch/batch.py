from tensordict import (
    NonTensorData,
    TensorDict,
    tensorclass,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor


@tensorclass(autocast=True)  # pyright: ignore[reportUntypedClassDecorator]
class BatchMeta:
    sample_idx: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    input_id: NonTensorData  # pyright: ignore[reportUninitializedInstanceVariable]


@tensorclass(autocast=True)  # pyright: ignore[reportUntypedClassDecorator]
class Batch:
    data: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    meta: BatchMeta  # pyright: ignore[reportUninitializedInstanceVariable]
