from tensordict import NonTensorStack, TensorClass, TensorDict


class BatchMeta(TensorClass, autocast=True):
    input_id: NonTensorStack


class Batch(TensorClass, autocast=True):
    data: TensorDict
    meta: BatchMeta | None = None
