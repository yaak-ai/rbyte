"""torchcodec decoding for the nero-arms disparity stream (data contract §21).

Two measured findings shape this module (§21.2):

1. FFV1 8-bit `gray` is **bit-exact** through torchcodec, which rbyte already
   uses for the mp4s -- so the depth stream reuses the `ImageFrameIndex` +
   `frame_index` machinery with no new decoder.
2. ⚠️ torchcodec **silently down-converts 16-bit video to uint8**. It does not
   raise; it returns plausible-looking wrong data (measured max error 58751).
   `_check_decodable` therefore refuses anything that is not 8-bit `gray`, and
   points at PyAV, which decodes 16-bit bit-exactly.

A lossy codec is refused for the same reason: libx264 on 8-bit disparity gave a
max error of 81 levels at 95 levels full scale.
"""

from collections.abc import Sequence
from typing import final, override

from pydantic import FilePath, InstanceOf, validate_call
from structlog import get_logger
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from rbyte.io.nero.calibration import NeroArmsCalibration, StereoCalibration
from rbyte.io.nero.disparity import (
    LOSSLESS_CODECS,
    PIXEL_FORMAT,
    DisparityOutput,
    disparity_to_depth,
)
from rbyte.types import TensorSource

__all__ = ["NeroArmsDisparityFrameSource"]

logger = get_logger(__name__)


@final
class NeroArmsDisparityFrameSource(TensorSource[int]):
    """Decode `{camera}_disparity.mkv` and optionally convert it to metric depth.

    One source == one output (`disparity` / `depth` / `valid`), because an rbyte
    stream is a single tensor. A depth-plus-mask config declares two streams over
    the same file.

    Deliberately exposes **no** transforms: resizing a disparity map scales
    `fx` but not the stored disparity values, so `fx * baseline / disparity`
    would come out wrong by the resize factor (see `StereoCalibration`).
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        source: FilePath | str,
        output: DisparityOutput = DisparityOutput.disparity,
        camera: str | None = None,
        calibration_path: FilePath | None = None,
        stereo: InstanceOf[StereoCalibration] | None = None,
        stream_index: int | None = None,
        num_ffmpeg_threads: int = 1,
        device: str | None = None,
    ) -> None:
        super().__init__()

        self._output = output
        self._decoder = VideoDecoder(
            source=source,
            stream_index=stream_index,
            dimension_order="NCHW",
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=device,
            seek_mode="exact",
        )
        self._check_decodable(str(source))
        self._stereo = (
            None
            if output is not DisparityOutput.depth
            else self._resolve_stereo(camera, calibration_path, stereo)
        )

    # -------------------------------------------------------------- validation

    def _check_decodable(self, source: str) -> None:
        metadata = self._decoder.metadata
        if (codec := metadata.codec) not in LOSSLESS_CODECS:
            logger.error(
                msg := "§21.2: disparity stream is not losslessly coded. A lossy "
                "codec is destruction, not degradation -- libx264 on 8-bit "
                "disparity measured a max error of 81 levels at 95 full scale.",
                source=source,
                codec=codec,
                expected=sorted(LOSSLESS_CODECS),
            )

            raise ValueError(msg)

        if (pixel_format := metadata.pixel_format) != PIXEL_FORMAT:
            logger.error(
                msg := "§21.2: disparity stream is not 8-bit gray. torchcodec "
                "SILENTLY down-converts 16-bit video to uint8 -- it does not "
                "raise, it returns wrong data. Decode this with PyAV instead, "
                "and check whether `subpixel` was left on.",
                source=source,
                pixel_format=pixel_format,
                expected=PIXEL_FORMAT,
            )

            raise ValueError(msg)

    def _resolve_stereo(
        self,
        camera: str | None,
        calibration_path: FilePath | None,
        stereo: StereoCalibration | None,
    ) -> StereoCalibration:
        if stereo is None:
            if calibration_path is None:
                logger.error(
                    msg := "§21.4: metric depth needs `stereo` or "
                    "`calibration_path`; pass `output=disparity` to ingest raw "
                    "disparity without calibration"
                )

                raise ValueError(msg)

            if camera is None:
                logger.error(msg := "`camera` is required with `calibration_path`")

                raise ValueError(msg)

            stereo = NeroArmsCalibration.from_path(calibration_path).stereo_for(camera)

        elif stereo.placeholder:
            logger.error(msg := "§21.4: `stereo` calibration is still PLACEHOLDER")

            raise ValueError(msg)

        metadata = self._decoder.metadata
        if (size := (metadata.width, metadata.height)) != stereo.image_size:
            logger.error(
                msg := "disparity frame size != `stereo.image_size`; `fx_mono` "
                "describes a different resolution, so every depth would be "
                "scaled wrong",
                size=size,
                image_size=stereo.image_size,
            )

            raise ValueError(msg)

        return stereo

    # ------------------------------------------------------------------ decode

    def _disparity(self, indexes: int | Sequence[int]) -> Tensor:
        """uint8 `(N, 1, H, W)` disparity levels, bit-exact (§21.2)."""
        match indexes:
            case Sequence():
                frames = self._decoder.get_frames_at(indices=list(indexes)).data

            case int():
                frames = self._decoder.get_frame_at(index=indexes).data.unsqueeze(0)

        # torchcodec hands back NCHW with a `gray` plane replicated across three
        # channels. Assert the replication rather than assuming it, then keep one.
        if frames.shape[-3] != 1:
            if not bool((frames == frames[..., :1, :, :]).all()):
                logger.error(
                    msg := "disparity frames are not single-channel gray",
                    shape=tuple(frames.shape),
                )

                raise ValueError(msg)

            frames = frames[..., :1, :, :]

        out = frames.contiguous()

        return out if isinstance(indexes, Sequence) else out.squeeze(0)

    @override
    def __getitem__(self, indexes: int | Sequence[int]) -> Tensor:
        disparity = self._disparity(indexes)
        match self._output:
            case DisparityOutput.disparity:
                return disparity

            case DisparityOutput.valid:
                return disparity > 0

            case DisparityOutput.depth:
                if self._stereo is None:
                    raise RuntimeError

                return disparity_to_depth(disparity, self._stereo)[0]

    @override
    def __len__(self) -> int:
        return self._decoder.metadata.num_frames or 0
