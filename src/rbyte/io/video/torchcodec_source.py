from collections.abc import Sequence
from contextlib import nullcontext
from enum import StrEnum, auto, unique
from typing import Annotated, final, override

import torch
from pydantic import AfterValidator, FilePath, InstanceOf, validate_call
from torch import Tensor
from torch.nn import Module
from torchcodec.decoders import VideoDecoder, set_cuda_backend
from torchcodec.transforms import DecoderTransform

from rbyte.types import TensorSource


@unique
class DimensionOrder(StrEnum):
    NCHW = "NCHW"
    NHWC = "NHWC"


@unique
class SeekMode(StrEnum):
    EXACT = auto()
    APPROXIMATE = auto()


@unique
class CudaBackend(StrEnum):
    BETA = auto()
    FFMPEG = auto()


@final
class TorchCodecFrameSource(TensorSource[int]):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        source: FilePath | str,
        stream_index: int | None = None,
        dimension_order: DimensionOrder = DimensionOrder.NCHW,
        num_ffmpeg_threads: int = 1,
        device: Annotated[str, AfterValidator(torch.device)]
        | InstanceOf[torch.device]
        | None = None,
        seek_mode: SeekMode = SeekMode.EXACT,
        transforms: Sequence[InstanceOf[DecoderTransform] | InstanceOf[Module]]
        | None = None,
        custom_frame_mappings: FilePath | None = None,
        cuda_backend: CudaBackend | None = None,
    ) -> None:
        super().__init__()

        if cuda_backend is None:
            match device, torch.get_default_device():
                case (torch.device(type="cuda"), _) | (None, torch.device(type="cuda")):
                    cuda_backend = CudaBackend.BETA

                case _:
                    cuda_backend = CudaBackend.FFMPEG

        with (
            set_cuda_backend(cuda_backend),
            (
                nullcontext()
                if custom_frame_mappings is None
                else custom_frame_mappings.open()
            ) as f_custom_frame_mappings,
        ):
            self._decoder = VideoDecoder(
                source=source,
                stream_index=stream_index,
                dimension_order=dimension_order.value,
                num_ffmpeg_threads=num_ffmpeg_threads,
                device=device,
                seek_mode=seek_mode.value,
                transforms=transforms,
                custom_frame_mappings=f_custom_frame_mappings,  # ty:ignore[invalid-argument-type]
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
