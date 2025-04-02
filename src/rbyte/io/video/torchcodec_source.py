from collections.abc import Iterable
from pathlib import Path
from typing import Literal, final, override

from pydantic import FilePath, validate_call
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from rbyte.io.base import TensorSource


@final
class TorchCodecFrameSource(TensorSource):
    @validate_call
    def __init__(
        self,
        path: FilePath,
        dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
        num_ffmpeg_threads: int = 1,
        device: str = "cpu",
        seek_mode: Literal["exact", "approximate"] = "exact",
    ) -> None:
        super().__init__()

        self._decoder = VideoDecoder(
            Path(path),
            dimension_order=dimension_order,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=device,
            seek_mode=seek_mode,
        )

    @override
    def __getitem__(self, indexes: int | Iterable[int]) -> Tensor:
        match indexes:
            case Iterable():
                frames = self._decoder.get_frames_at(indices=list(indexes)).data
            case _:
                frames = self._decoder.get_frame_at(index=indexes).data

        return frames

    @override
    def __len__(self) -> int:
        return self._decoder.metadata.num_frames  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportUnknownMemberType]
