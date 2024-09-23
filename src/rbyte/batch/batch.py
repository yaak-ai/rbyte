from jaxtyping import Int
from tensordict import NonTensorData, TensorDict, tensorclass
from torch import Tensor


@tensorclass  # pyright: ignore[reportUntypedClassDecorator, reportArgumentType, reportCallIssue]
class BatchMeta:
    sample_idx: Int[Tensor, "b 1"]  # pyright: ignore[reportUninitializedInstanceVariable]
    input_id: NonTensorData  # pyright: ignore[reportUninitializedInstanceVariable]


@tensorclass  # pyright: ignore[reportUntypedClassDecorator, reportArgumentType, reportCallIssue]
class Batch:
    meta: BatchMeta  # pyright: ignore[reportUninitializedInstanceVariable]
    frame: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    table: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
