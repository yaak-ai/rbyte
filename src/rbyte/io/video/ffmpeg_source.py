from collections.abc import Iterable
from pathlib import Path
from typing import final, override

import torch
from pydantic import FilePath, NonNegativeInt, validate_call
from torch import Tensor
from video_reader import (
    PyVideoReader,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)

from rbyte.io.base import TensorSource


@final
class FfmpegFrameSource(TensorSource):
    @validate_call
    def __init__(
        self,
        path: FilePath,
        threads: NonNegativeInt | None = None,
        resize_shorter_side: NonNegativeInt | None = None,
        device: str | None = None,
        filter: str | None = None,
    ) -> None:
        super().__init__()

        self._reader: PyVideoReader = PyVideoReader(
            filename=Path(path).resolve().as_posix(),
            threads=threads,
            resize_shorter_side=resize_shorter_side,
            device=device,
            filter=filter,
        )

    @override
    def __getitem__(self, indexes: int | Iterable[int]) -> Tensor:
        match indexes:
            case Iterable():
                array = self._reader.get_batch(indexes)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

            case _:
                array = self._reader.get_batch([indexes])[0]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        return torch.from_numpy(array)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    @override
    def __len__(self) -> int:
        return int(self._reader.get_info()["frame_count"])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
