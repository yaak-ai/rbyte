from collections.abc import Iterable
from pathlib import Path
from typing import cast, override

import numpy.typing as npt
import torch
from jaxtyping import UInt8
from pydantic import FilePath, NonNegativeInt, validate_call
from torch import Tensor
from video_reader import (
    PyVideoReader,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)

from rbyte.io.base import TensorSource


class FfmpegFrameSource(TensorSource):
    @validate_call
    def __init__(
        self,
        path: FilePath,
        threads: NonNegativeInt | None = None,
        resize_shorter_side: NonNegativeInt | None = None,
    ) -> None:
        super().__init__()

        self._reader: PyVideoReader = PyVideoReader(
            filename=Path(path).resolve().as_posix(),
            threads=threads,
            resize_shorter_side=resize_shorter_side,
        )

    @override
    def __getitem__(self, indexes: Iterable[int]) -> UInt8[Tensor, "b h w c"]:
        batch = cast(npt.ArrayLike, self._reader.get_batch(indexes))  # pyright: ignore[reportUnknownMemberType]

        return torch.from_numpy(batch)  # pyright: ignore[reportUnknownMemberType]

    @override
    def __len__(self) -> int:
        return int(self._reader.get_info()["frame_count"])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
