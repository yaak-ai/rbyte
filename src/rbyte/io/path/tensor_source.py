from collections.abc import Callable, Iterable
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any, override

import numpy.typing as npt
import torch
from jaxtyping import UInt8
from pydantic import validate_call
from torch import Tensor

from rbyte.io.base import TensorSource


class PathTensorSource(TensorSource):
    @validate_call
    def __init__(
        self, path: PathLike[str], decoder: Callable[[bytes], npt.ArrayLike]
    ) -> None:
        super().__init__()

        self._path: Path = Path(path)
        self._decoder: Callable[[bytes], npt.ArrayLike] = decoder

    @cached_property
    def _path_posix(self) -> str:
        return self._path.resolve().as_posix()

    def _decode(self, path: str) -> npt.ArrayLike:
        with Path(path).open("rb") as f:
            return self._decoder(f.read())

    @override
    def __getitem__(self, indexes: Iterable[Any]) -> UInt8[Tensor, "b h w c"]:
        paths = map(self._path_posix.format, indexes)
        arrays = map(self._decode, paths)
        tensors = map(torch.from_numpy, arrays)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        return torch.stack(list(tensors))

    @override
    def __len__(self) -> int:
        raise NotImplementedError
