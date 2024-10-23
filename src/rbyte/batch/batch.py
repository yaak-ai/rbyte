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
    meta: BatchMeta  # pyright: ignore[reportUninitializedInstanceVariable]
    frame: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    table: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
