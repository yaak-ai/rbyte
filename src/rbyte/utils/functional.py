from collections.abc import Sequence

import more_itertools as mit
import torch
import torch.nn.functional as F  # noqa: N812
from jaxtyping import Float
from torch import Tensor


def pad_dim(
    input: Float[Tensor, "..."],  # noqa: A002
    *,
    pad: tuple[int, int],
    dim: int,
    mode: str = "constant",
    value: float | None = None,
) -> Float[Tensor, "..."]:
    _pad = [(0, 0) for _ in input.shape]
    _pad[dim] = pad
    _pad = list(mit.flatten(reversed(_pad)))

    return F.pad(input, _pad, mode=mode, value=value)


def pad_sequence(
    sequences: Sequence[Float[Tensor, "..."]], dim: int, value: float = 0.0
) -> Float[Tensor, "..."]:
    max_length = max(sequence.shape[dim] for sequence in sequences)

    padded = (
        pad_dim(
            sequence,
            pad=(0, max_length - sequence.shape[dim]),
            dim=dim,
            mode="constant",
            value=value,
        )
        for sequence in sequences
    )

    return torch.stack(list(padded))
