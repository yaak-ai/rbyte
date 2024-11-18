from collections.abc import Iterable, Mapping
from functools import cached_property
from itertools import pairwise
from typing import Annotated, override

import more_itertools as mit
import python_vali as vali
import torch
from jaxtyping import Shaped
from pydantic import BeforeValidator, FilePath, NonNegativeInt, validate_call
from structlog import get_logger
from torch import Tensor

from rbyte.config.base import BaseModel
from rbyte.io.base import TensorSource

logger = get_logger(__name__)

PixelFormat = Annotated[
    vali.PixelFormat,
    BeforeValidator(
        lambda x: x if isinstance(x, vali.PixelFormat) else getattr(vali.PixelFormat, x)
    ),
]


class ValiGpuFrameSource(TensorSource):
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self,
        path: FilePath,
        gpu_id: NonNegativeInt = 0,
        pixel_format_chain: tuple[PixelFormat, ...] = (
            vali.PixelFormat.RGB,
            vali.PixelFormat.RGB_PLANAR,
        ),
    ) -> None:
        super().__init__()

        self._gpu_id: int = gpu_id

        self._decoder: vali.PyDecoder = vali.PyDecoder(
            input=path.resolve().as_posix(), opts={}, gpu_id=self._gpu_id
        )

        self._pixel_format_chain: tuple[PixelFormat, ...] = (
            (self._decoder.Format, *pixel_format_chain)
            if mit.first(pixel_format_chain, default=None) != self._decoder.Format
            else pixel_format_chain
        )

    @cached_property
    def _surface_converters(
        self,
    ) -> Mapping[tuple[vali.PixelFormat, vali.PixelFormat], vali.PySurfaceConverter]:
        return {
            (src_format, dst_format): vali.PySurfaceConverter(
                src_format=src_format, dst_format=dst_format, gpu_id=self._gpu_id
            )
            for src_format, dst_format in pairwise(self._pixel_format_chain)
        }

    @cached_property
    def _surfaces(self) -> Mapping[vali.PixelFormat, vali.Surface]:
        return {
            pixel_format: vali.Surface.Make(
                format=pixel_format,
                width=self._decoder.Width,
                height=self._decoder.Height,
                gpu_id=self._gpu_id,
            )
            for pixel_format in self._pixel_format_chain
        }

    def _read_frame(
        self, index: int
    ) -> Shaped[Tensor, "c h w"] | Shaped[Tensor, "h w c"]:
        seek_ctx = vali.SeekContext(seek_frame=index)
        success, details = self._decoder.DecodeSingleSurface(  # pyright: ignore[reportUnknownMemberType]
            self._surfaces[self._decoder.Format], seek_ctx
        )
        if not success:
            logger.error(msg := "failed to decode surface", details=details)

            raise RuntimeError(msg)

        for (src_format, dst_format), converter in self._surface_converters.items():
            success, details = converter.Run(  # pyright: ignore[reportUnknownMemberType]
                (src := self._surfaces[src_format]), (dst := self._surfaces[dst_format])
            )
            if not success:
                logger.error(
                    msg := "failed to convert surface",
                    src=src,
                    dst=dst,
                    details=details,
                )

                raise RuntimeError(msg)

        surface = self._surfaces[self._pixel_format_chain[-1]]

        return torch.from_dlpack(surface).clone().detach()  # pyright: ignore[reportPrivateImportUsage]

    @override
    def __getitem__(
        self, indexes: Iterable[int]
    ) -> Shaped[Tensor, "b h w c"] | Shaped[Tensor, "b c h w"]:
        return torch.stack([self._read_frame(index) for index in indexes])

    @override
    def __len__(self) -> int:
        return self._decoder.NumFrames
