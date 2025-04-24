from collections.abc import Sequence
from typing import Literal, final, override

from pydantic import FilePath, validate_call
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from rbyte.io.base import TensorSource


@final
class TorchCodecFrameSource(TensorSource[int]):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        source: FilePath | str,
        stream_index: int | None = None,
        dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
        num_ffmpeg_threads: int = 1,
        device: str | None = "cpu",
        seek_mode: Literal["exact", "approximate"] = "exact",
    ) -> None:
        super().__init__()

        self._decoder = VideoDecoder(
            source=source,
            stream_index=stream_index,
            dimension_order=dimension_order,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=device,
            seek_mode=seek_mode,
        )

    @override
    def __getitem__(self, indexes: int | Sequence[int]) -> Tensor:
        match indexes:
            case Sequence():
                return self._decoder.get_frames_at(indices=indexes).data  # pyright: ignore[reportArgumentType]

            case int():
                return self._decoder.get_frame_at(index=indexes).data

    @override
    def __len__(self) -> int:
        return self._decoder.metadata.num_frames or 0
