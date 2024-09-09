from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import override

import numpy.typing as npt
import parse
import torch
from jaxtyping import UInt8
from pydantic import validate_call
from torch import Tensor

from rbyte.io.frame.base import FrameReader


class DirectoryFrameReader(FrameReader):
    @validate_call
    def __init__(
        self, path: PathLike[str], frame_decoder: Callable[[bytes], npt.ArrayLike]
    ) -> None:
        super().__init__()

        self._path = Path(path)
        self._frame_decoder = frame_decoder

    @cached_property
    def _path_posix(self) -> str:
        return self._path.as_posix()

    def _decode(self, path: str) -> npt.ArrayLike:
        with Path(path).open("rb") as f:
            return self._frame_decoder(f.read())

    @override
    def read(self, indexes: Iterable[int]) -> UInt8[Tensor, "b h w c"]:
        paths = map(self._path_posix.format, indexes)
        frames_np = map(self._decode, paths)
        frames_tch = map(torch.from_numpy, frames_np)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        return torch.stack(list(frames_tch))

    @override
    def get_available_indexes(self) -> Sequence[int]:
        parser = parse.compile(self._path.name)  # pyright: ignore[reportUnknownMemberType]
        filenames = (path.name for path in self._path.parent.iterdir())
        return [res[0] for res in map(parser.parse, filenames) if res]  # pyright: ignore[reportUnknownVariableType, reportIndexIssue, reportUnknownArgumentType, reportUnknownMemberType]
