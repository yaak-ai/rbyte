from jaxtyping import Int
from tensordict import NonTensorData, TensorDict, tensorclass
from torch import Tensor


@tensorclass  # pyright: ignore[reportArgumentType]
class BatchMeta:
    sample_idx: Int[Tensor, "b 1"]  # pyright: ignore[reportUninitializedInstanceVariable]
    input_id: NonTensorData  # pyright: ignore[reportGeneralTypeIssues, reportUninitializedInstanceVariable]


@tensorclass  # pyright: ignore[reportArgumentType]
class Batch:
    meta: BatchMeta  # pyright: ignore[reportUninitializedInstanceVariable, reportGeneralTypeIssues]
    frame: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    table: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
