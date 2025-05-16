from collections.abc import Sequence
from functools import cached_property
from typing import Annotated, final, override

import av
import m3u8
import polars as pl
import torch
from pydantic import BeforeValidator, InstanceOf, validate_call
from s3pathlib import S3Path
from structlog import get_logger
from torch import Tensor

from rbyte.io.base import TensorSource

logger = get_logger(__name__)


@final
class HlsFrameSource(TensorSource[int]):
    @validate_call
    def __init__(
        self,
        *,
        path: Annotated[InstanceOf[S3Path], BeforeValidator(S3Path.from_s3_uri)],
        fps: int,
    ) -> None:
        super().__init__()

        self._path = path
        self._fps = fps

    @cached_property
    def _frame_index(self) -> pl.DataFrame:
        with self._path.open("r") as f:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            playlist = m3u8.loads(f.read())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        frame_data: list[tuple[int, str | None, int]] = []
        total_frames: int = 0

        for segment in playlist.segments:
            if segment.uri is None or segment.duration is None:
                logger.error(msg := "invalid segment", segment=segment)
                raise RuntimeError(msg)

            num_frames = round(segment.duration * self._fps)
            frame_data.extend(
                (frame_idx, segment.uri, total_frames)
                for frame_idx in range(total_frames, total_frames + num_frames)
            )
            total_frames += num_frames

        return pl.DataFrame(
            frame_data,
            schema=["frame_idx", "segment_uri", "total_frames"],
            orient="row",
        )

    @override
    def __getitem__(self, indexes: int | Sequence[int]) -> Tensor:
        match indexes:
            case int():
                return self._get([indexes])[0]

            case _:
                return torch.stack(self._get(indexes))

    def _get(self, indexes: Sequence[int]) -> list[Tensor]:
        outputs: list[Tensor] = []
        rows = self._frame_index.filter(pl.col("frame_idx").is_in(indexes))

        for frame_idx, segment_uri, total_frames in rows.iter_rows():
            with (
                (self._path.parent / segment_uri).open("rb") as segment,
                av.open(segment) as container,
            ):
                stream = container.streams.video[0]

                if (
                    stream.start_time is None
                    or stream.time_base is None
                    or stream.average_rate is None
                ):
                    logger.debug(msg := "invalid stream", stream=stream)
                    raise RuntimeError(msg)

                # Calculate key frame and seek position
                avg_rate = int(stream.average_rate)
                key_frame_offset = (frame_idx - total_frames) // avg_rate * avg_rate
                seek_pts = int(
                    stream.start_time + (key_frame_offset / avg_rate) / stream.time_base
                )
                # Number of frames to skip after seeking
                frames_to_skip = frame_idx - total_frames - key_frame_offset

                container.seek(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnusedCallResult]
                    seek_pts, any_frame=False, backward=True, stream=stream
                )

                for _ in range(frames_to_skip):
                    next(container.decode(stream))  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnusedCallResult, reportUnknownArgumentType]

                frame = next(container.decode(stream))  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownArgumentType]

                outputs.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))  # pyright: ignore[reportUnknownMemberType]

        return outputs

    @override
    def __len__(self) -> int:
        return len(self._frame_index)
