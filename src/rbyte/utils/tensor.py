from collections.abc import Sequence

import more_itertools as mit
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def pad_dim(
    input: Tensor,
    *,
    pad: tuple[int, int],
    dim: int,
    mode: str = "constant",
    value: float | None = None,
) -> Tensor:
    pad_ = [(0, 0) for _ in input.shape]
    pad_[dim] = pad
    pad_ = list(mit.flatten(reversed(pad_)))

    return F.pad(input, pad_, mode=mode, value=value)


def pad_sequence(sequences: Sequence[Tensor], dim: int, value: float = 0.0) -> Tensor:
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
