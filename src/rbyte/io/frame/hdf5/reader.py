from collections.abc import Iterable, Sequence
from typing import cast, override

import h5py
import torch
from jaxtyping import UInt8
from pydantic import FilePath, validate_call
from torch import Tensor

from rbyte.io.frame.base import FrameReader


class Hdf5FrameReader(FrameReader):
    @validate_call
    def __init__(self, path: FilePath, key: str) -> None:
        file = h5py.File(path)
        self._dataset = cast(h5py.Dataset, file[key])

    @override
    def read(self, indexes: Iterable[int]) -> UInt8[Tensor, "b h w c"]:
        return torch.from_numpy(self._dataset[indexes])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    @override
    def get_available_indexes(self) -> Sequence[int]:
        return range(len(self._dataset))
