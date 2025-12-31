from collections.abc import Callable, Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import final, override

import numpy.typing as npt
import torch
from pydantic import validate_call
from torch import Tensor

from rbyte.types import TensorSource


@final
class PathTensorSource(TensorSource[object]):
    @validate_call
    def __init__(
        self,
        *,
        path: PathLike[str],
        decoder: Callable[[bytes], npt.ArrayLike],
        index_transform: Callable[..., object] | None = None,
    ) -> None:
        super().__init__()

        self._path = Path(path)
        self._decoder = decoder
        self._index_transform = index_transform

    @cached_property
    def _path_posix(self) -> str:
        return self._path.resolve().as_posix()

    def _decode(self, path: str) -> npt.ArrayLike:
        return self._decoder(Path(path).read_bytes())

    def _getitem(self, index: object) -> Tensor:
        if self._index_transform is not None:
            index = self._index_transform(index)

        path = self._path_posix.format(index)
        array = self._decode(path)

        return torch.from_numpy(array)

    @override
    def __getitem__(self, indexes: object | Sequence[object]) -> Tensor:
        match indexes:
            case Sequence():
                return torch.stack([self._getitem(i) for i in indexes])

            case _:
                return self._getitem(indexes)

    @override
    def __len__(self) -> int:
        raise NotImplementedError
