from collections.abc import Callable, Iterable, Sequence
from functools import partial
from os import PathLike
from pathlib import Path
from typing import override

import torch
import video_reader as vr
from jaxtyping import UInt8
from pydantic import NonNegativeInt, validate_call
from torch import Tensor

from rbyte.io.frame.base import FrameReader


class VideoFrameReader(FrameReader):
    @validate_call
    def __init__(
        self,
        path: PathLike[str],
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
        num_frames, *_ = vr.get_shape(self._path)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportUnknownMemberType]

        return range(num_frames)  # pyright: ignore[reportUnknownArgumentType]
