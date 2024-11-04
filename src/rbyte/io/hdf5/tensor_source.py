from collections.abc import Iterable
from typing import cast, override

import torch
from h5py import Dataset, File
from jaxtyping import UInt8
from pydantic import FilePath, validate_call
from torch import Tensor

from rbyte.io.base import TensorSource


class Hdf5TensorSource(TensorSource):
    @validate_call
    def __init__(self, path: FilePath, key: str) -> None:
        file = File(path)
        self._dataset: Dataset = cast(Dataset, file[key])

    @override
    def __getitem__(self, indexes: Iterable[int]) -> UInt8[Tensor, "b h w c"]:
        return torch.from_numpy(self._dataset[indexes])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    @override
    def __len__(self) -> int:
        return len(self._dataset)
