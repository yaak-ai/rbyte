from collections.abc import Callable, Iterable
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import final, override

import numpy.typing as npt
import torch
from pydantic import validate_call
from torch import Tensor

from rbyte.io.base import TensorSource


@final
class PathTensorSource(TensorSource):
    @validate_call
    def __init__(
        self, path: PathLike[str], decoder: Callable[[bytes], npt.ArrayLike]
    ) -> None:
        super().__init__()

        self._path = Path(path)
        self._decoder = decoder

    @cached_property
    def _path_posix(self) -> str:
        return self._path.resolve().as_posix()

    def _decode(self, path: str) -> npt.ArrayLike:
        with Path(path).open("rb") as f:
            return self._decoder(f.read())

    def _getitem(self, index: object) -> Tensor:
        path = self._path_posix.format(index)
        array = self._decode(path)
        return torch.from_numpy(array)  # pyright: ignore[reportUnknownMemberType]

    @override
    def __getitem__(self, indexes: object | Iterable[object]) -> Tensor:
        match indexes:
            case Iterable():
                tensors = map(self._getitem, indexes)  # pyright: ignore[reportUnknownArgumentType]

                return torch.stack(list(tensors))

            case _:
                return self._getitem(indexes)

    @override
    def __len__(self) -> int:
        raise NotImplementedError
