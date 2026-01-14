from collections.abc import Sequence
from typing import cast, final

import torch
from h5py import Dataset, File
from pydantic import FilePath, validate_call
from torch import Tensor
from typing_extensions import override

from rbyte.types import TensorSource


@final
class Hdf5TensorSource(TensorSource[int]):
    @validate_call
    def __init__(self, path: FilePath, key: str) -> None:
        self._dataset = cast(Dataset, File(path)[key])

    @override
    def __getitem__(self, indexes: int | Sequence[int]) -> Tensor:
        return torch.from_numpy(self._dataset[indexes])

    @override
    def __len__(self) -> int:
        return len(self._dataset)
