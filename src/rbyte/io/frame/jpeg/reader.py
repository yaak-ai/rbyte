from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any, override

import parse
import torch
from jaxtyping import UInt8
from simplejpeg import decode_jpeg  # pyright: ignore[reportUnknownVariableType]
from torch import Tensor

from rbyte.io.frame.base import FrameReader


class JpegFrameReader(FrameReader):
    def __init__(
        self, path: PathLike[str], decode_kwargs: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__()

        self.path = Path(path)
        self.decode_kwargs = decode_kwargs or {}

    @cached_property
    def _path_posix(self) -> str:
        return self.path.as_posix()

    def _decode_jpeg(self, path: str) -> object:
        with Path(path).open("rb") as f:
            return decode_jpeg(f.read(), **self.decode_kwargs)  # pyright: ignore[reportUnknownVariableType]

    @override
    def read(self, idxs: Iterable[int]) -> UInt8[Tensor, "b h w c"]:
        paths = map(self._path_posix.format, idxs)
        frames_np = map(self._decode_jpeg, paths)
        frames_tch = map(torch.from_numpy, frames_np)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        return torch.stack(list(frames_tch))

    @override
    def get_available_indices(self) -> Sequence[int]:
        parser = parse.compile(self.path.name)  # pyright: ignore[reportUnknownMemberType]
        filenames = (path.name for path in self.path.parent.iterdir())
        return [res[0] for res in map(parser.parse, filenames) if res]  # pyright: ignore[reportUnknownVariableType, reportIndexIssue, reportUnknownArgumentType, reportUnknownMemberType]
