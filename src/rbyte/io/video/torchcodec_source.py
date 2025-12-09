from collections.abc import Sequence
from typing import Annotated, Literal, final, override

import torch
from pydantic import AfterValidator, FilePath, InstanceOf, validate_call
from torch import Tensor
from torchcodec.decoders import VideoDecoder, set_cuda_backend

from rbyte.types import TensorSource


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
        device: Annotated[str, AfterValidator(torch.device)]
        | InstanceOf[torch.device] = "cpu",
        seek_mode: Literal["exact", "approximate"] = "exact",
        cuda_backend: Literal["ffmpeg", "beta"] | None = None,
    ) -> None:
        super().__init__()

        if cuda_backend is None:
            match device:
                case torch.device(type="cuda"):
                    cuda_backend = "beta"

                case _:
                    cuda_backend = "ffmpeg"

        with set_cuda_backend(cuda_backend):
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
                return self._decoder.get_frames_at(indices=list(indexes)).data

            case int():
                return self._decoder.get_frame_at(index=indexes).data

            case _:
                raise ValueError

    @override
    def __len__(self) -> int:
        return self._decoder.metadata.num_frames or 0
