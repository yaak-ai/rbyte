from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import override

import torch
import video_reader as vr
from jaxtyping import UInt8
from pydantic import FilePath, NonNegativeInt, validate_call
from torch import Tensor

from rbyte.io.frame.base import FrameReader


class FfmpegFrameReader(FrameReader):
    @validate_call
    def __init__(
        self,
        path: FilePath,
        threads: NonNegativeInt | None = None,
        resize_shorter_side: NonNegativeInt | None = None,
        with_fallback: bool | None = None,  # noqa: FBT001
    ) -> None:
        super().__init__()
        self._path = Path(path).resolve().as_posix()

        self._get_batch: Callable[[str, Iterable[int]], UInt8[Tensor, "b h w c"]] = (
            partial(
                vr.get_batch,  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                threads=threads,
                resize_shorter_side=resize_shorter_side,
                with_fallback=with_fallback,
            )
        )

    @override
    def read(self, indexes: Iterable[int]) -> UInt8[Tensor, "b h w c"]:
        batch = self._get_batch(self._path, indexes)

        return torch.from_numpy(batch)  # pyright: ignore[reportUnknownMemberType]

    @override
    def get_available_indexes(self) -> Sequence[int]:
        num_frames = int(vr.get_info(self._path)["frame_count"])  # pyright: ignore[reportAttributeAccessIssue, reportUnknownArgumentType, reportUnknownMemberType]

        return range(num_frames)
