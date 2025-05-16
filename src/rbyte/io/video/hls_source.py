from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, final, override

import av
import fsspec
import m3u8
import polars as pl
import torch
from av.container.input import InputContainer
from pydantic import FilePath
from s3pathlib import S3Path

from rbyte.io.base import TensorSource

if TYPE_CHECKING:
    from av.video.frame import VideoFrame, VideoStream
    from numpy import ndarray


@final
class HLSFrameSource(TensorSource[int]):
    def __init__(
        self,
        source: FilePath | str,
        dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
        filesystem: str = "s3",
        fps: int = 30,
    ) -> None:
        super().__init__()
        self.source: FilePath | str = source
        self.dimension_order: Literal["NCHW", "NHWC"] = dimension_order
        self.fs = fsspec.filesystem(filesystem)
        self.fps: int = fps
        self.segments = self._load_segments()
        self.frame_index_table = self._build_frame_index_table()
        self.num_frames = len(self.frame_index_table)

    def _load_segments(self) -> m3u8.model.SegmentList:  # pyright: ignore[reportUnknownMemberType]
        with self.fs.open(self.source, "r") as f:
            playlist = m3u8.loads(f.read())
        return playlist.segments

    def _build_frame_index_table(self) -> pl.DataFrame:
        frame_data: list = []
        total_frames: int = 0

        for segment in self.segments:
            num_frames = round(segment.duration * self.fps)
            frame_data.extend(
                (frame_idx, segment.uri, total_frames)
                for frame_idx in range(total_frames, total_frames + num_frames)
            )
            total_frames += num_frames

        return pl.DataFrame(
            frame_data, schema=["frame_idx", "uri", "past_frames"], orient="row"
        )

    def _get_uri_path(self, chunk: str) -> S3Path:
        parent = S3Path(self.source).parent
        return parent.joinpath(chunk).uri

    def _get_container(self, uri: str) -> InputContainer:
        fileobj = self.fs.open(uri, mode="rb")
        return av.open(fileobj)

    @override
    def __getitem__(self, indexes: int | Sequence[int]) -> torch.Tensor:
        if isinstance(indexes, int):
            indexes = [indexes]

        outputs: list[torch.Tensor] = []
        rows = self.frame_index_table.filter(pl.col("frame_idx").is_in(indexes))

        for row in rows.iter_rows(named=True):
            frame_idx: int = row["frame_idx"]
            uri: str = row["uri"]
            past_frames: int = row["past_frames"]

            container: InputContainer = self._get_container(self._get_uri_path(uri))
            stream: VideoStream = container.streams.video[0]

            # Calculate key frame and seek position
            avg_rate: int = int(stream.average_rate)
            key_frame_offset: int = (frame_idx - past_frames) // avg_rate * avg_rate
            seek_pts: int = int(
                stream.start_time + (key_frame_offset / avg_rate) / stream.time_base
            )

            # Number of frames to skip after seeking
            frames_to_skip: int = frame_idx - past_frames - key_frame_offset

            # https://pyav.org/docs/stable/api/container.html#av.container.InputContainer.seek
            container.seek(seek_pts, any_frame=False, backward=True, stream=stream)

            for _ in range(frames_to_skip):
                next(container.decode(stream))

            frame: VideoFrame = next(container.decode(stream))
            img: ndarray = frame.to_ndarray(format="rgb24")  # pyright: ignore[reportMissingTypeArgument]
            outputs.append(torch.tensor(img, dtype=torch.uint8))

            container.close()

        stacked = torch.stack(outputs)
        return (
            stacked.permute(0, 3, 1, 2) if self.dimension_order == "NCHW" else stacked
        )

    @override
    def __len__(self) -> int:
        return self.num_frames
