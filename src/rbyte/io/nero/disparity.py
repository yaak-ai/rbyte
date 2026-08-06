"""nero-arms depth stream: 8-bit disparity -> metric depth (data contract §21).

The recorder writes **disparity**, not depth, as 8-bit `gray` FFV1 in MKV
(`base_disparity.mkv`), overhead camera only. §21.1: disparity is exactly
representable in 8 bits as long as `subpixel` is off, which halves the data
before any codec runs and defers the metric conversion to here, where the
calibration lives.

This module is the calibration/declaration half and deliberately does **not**
import torchcodec, so `rbyte.io.nero` stays importable without the `video`
extra. The decoder lives in `disparity_source.py`.
"""

from enum import StrEnum, auto, unique
from typing import Final, Self

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
from structlog import get_logger
from torch import Tensor

from rbyte.io.nero.calibration import StereoCalibration, check_disparity_mode

__all__ = [
    "DISPARITY_METADATA_PREFIX",
    "DISPARITY_TOPIC_PREFIX",
    "LOSSLESS_CODECS",
    "PIXEL_FORMAT",
    "DisparityDeclaration",
    "DisparityOutput",
    "disparity_to_depth",
]

logger = get_logger(__name__)

#: §21.3: same `ImageFrameIndex` schema as the cameras, own `frame_index`.
DISPARITY_TOPIC_PREFIX: Final = "observation.disparity."
#: §21.3 episode metadata record, alongside the existing `camera.{name}` ones.
DISPARITY_METADATA_PREFIX: Final = "disparity."

#: §21.2: the only pixel format that survives torchcodec bit-exactly.
PIXEL_FORMAT: Final = "gray"
#: §21.2: never a lossy codec.
LOSSLESS_CODECS: Final = frozenset({"ffv1"})


@unique
class DisparityOutput(StrEnum):
    """What a `NeroArmsDisparityFrameSource` emits per frame."""

    #: raw uint8 disparity levels; needs no calibration
    disparity = auto()
    #: float32 metres, 0.0 where invalid -- see `valid`
    depth = auto()
    #: bool, False where `disparity == 0` (§21.4: invalid, not zero distance)
    valid = auto()


class DisparityDeclaration(BaseModel):
    """§21.3 episode metadata for the disparity stream.

    Per §17.1 these are **declared**, never inferred from the pixels: a consumer
    that assumes 95 when extended disparity was on computes every depth wrong by
    2x, and there is nothing in the frames that says which was used.
    """

    max_disparity: PositiveInt
    subpixel: bool
    extended_disparity: bool
    width: PositiveInt | None = None
    height: PositiveInt | None = None
    mono_sockets: str | None = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        check_disparity_mode(
            max_disparity=self.max_disparity,
            subpixel=self.subpixel,
            extended_disparity=self.extended_disparity,
            source="episode metadata",
        )

        return self

    @classmethod
    def from_mcap_metadata(cls, metadata: dict[str, str]) -> Self:
        """MCAP metadata records are `dict[str, str]`; booleans arrive as text."""

        def parse(value: str) -> bool:
            match value.strip().lower():
                case "true" | "1" | "yes":
                    return True

                case "false" | "0" | "no":
                    return False

                case _:
                    logger.error(
                        msg := "§21.3: undecodable boolean in episode metadata",
                        value=value,
                    )

                    raise ValueError(msg)

        return cls.model_validate(
            metadata
            | {
                key: parse(metadata[key])
                for key in ("subpixel", "extended_disparity")
                if key in metadata
            }
        )

    def check_against(self, stereo: StereoCalibration, *, camera: str) -> None:
        """The per-episode declaration wins; a disagreement is a hard failure."""
        mismatched = {
            field: (declared, calibrated)
            for field in ("max_disparity", "subpixel", "extended_disparity")
            if (declared := getattr(self, field))
            != (calibrated := getattr(stereo, field))
        }
        if self.width is not None and self.height is not None:
            size = (self.width, self.height)
            if size != stereo.image_size:
                mismatched["image_size"] = (size, stereo.image_size)

        if mismatched:
            logger.error(
                msg := "§21.3: episode metadata disagrees with `stereo` calibration",
                camera=camera,
                mismatched=mismatched,
            )

            raise ValueError(msg)


def disparity_to_depth(
    disparity: Tensor, stereo: StereoCalibration
) -> tuple[Tensor, Tensor]:
    """§21.4: `(depth_m, valid)` from uint8 disparity levels.

    `depth_m = fx_mono_px * baseline_m / disparity`, with `disparity == 0`
    meaning **invalid, not zero distance**. The denominator is clamped *before*
    the division, so an invalid pixel can never produce `inf` (which would turn
    into `nan` the moment anything multiplies it by the mask) and never becomes a
    0 m reading either -- it is 0.0 *and* masked out, and consumers must use the
    mask.
    """
    valid = disparity > 0
    safe = disparity.to(torch.float32).clamp(min=1.0)
    depth = torch.where(
        valid,
        stereo.fx_mono * stereo.baseline_m / safe,
        torch.zeros((), dtype=safe.dtype),
    )

    return depth, valid
