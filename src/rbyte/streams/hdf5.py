from collections.abc import Sequence
from typing import final, override

import torch
from h5py import Dataset, File
from pydantic import FilePath, validate_call
from torch import Tensor

from rbyte.streams.base import StreamSource


@final
class Hdf5Source(StreamSource[int]):
    @validate_call
    def __init__(self, path: FilePath, key: str) -> None:
        self._dataset: Dataset = File(path)[key]

    @override
    def __getitem__(self, indexes: int | Sequence[int]) -> Tensor:
        return torch.from_numpy(self._dataset[indexes])
