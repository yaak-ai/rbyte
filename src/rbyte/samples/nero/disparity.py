"""nero-arms depth stream: disparity -> metric depth (data contract §21).

The recorder writes **disparity**, not depth, as FFV1 in MKV
(`base_disparity.mkv`), overhead camera only, and it may be **8-bit or 16-bit**
(§21.16). 16-bit with subpixel on is the primary form: §21.14 measured that
integer disparity cannot resolve the manipulated object at the rig's working
distance -- a 2 cm cube at 0.64 m spans 1.1 disparity levels -- while at 5
fractional bits the same delta spans ~35 units.

**Which bit depth arrived decides which decoder is legal**, and that is the whole
reason the recorder declares it. FFV1 `gray16le` round-trips bit-exact through
PyAV; torchcodec **silently** returns uint8 for the same file (measured max error
58751, no exception). So `disparity_source.py` routes on the container's pixel
format and the torchcodec path refuses 16-bit outright.

This module is the calibration/declaration half and deliberately imports neither
decoder, so `rbyte.samples.nero` stays importable without the `video` or `pyav`
extras.
"""

from enum import StrEnum, auto, unique
from typing import Final, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
from structlog import get_logger
from torch import Tensor

from rbyte.samples.nero.calibration import StereoCalibration, check_disparity_mode

__all__ = [
    "BIT_DEPTH_PYAV_ONLY",
    "DISPARITY_METADATA_PREFIX",
    "DISPARITY_PIXEL_FORMATS",
    "DISPARITY_TOPIC_PREFIX",
    "LOSSLESS_CODECS",
    "PIXEL_FORMAT",
    "DisparityDeclaration",
    "DisparityOutput",
    "check_store_against_mode",
    "disparity_to_depth",
]

logger = get_logger(__name__)

#: §21.3: same `ImageFrameIndex` schema as the cameras, own `frame_index`.
DISPARITY_TOPIC_PREFIX: Final = "observation.disparity."
#: §21.3 episode metadata record, alongside the existing `camera.{name}` ones.
DISPARITY_METADATA_PREFIX: Final = "disparity."

#: §21.2: the only pixel format that survives **torchcodec** bit-exactly.
PIXEL_FORMAT: Final = "gray"
#: §21.16: the two stores the recorder may write, and the bit depth of each.
#: `gray16le` is bit-exact through PyAV and *silently wrong* through torchcodec,
#: which is what makes this mapping the routing key rather than a description.
DISPARITY_PIXEL_FORMATS: Final[dict[str, int]] = {"gray": 8, "gray16le": 16}
#: §21.16: the store that **only** PyAV can read. torchcodec returns uint8 for
#: it without raising, so this number is a routing threshold, not a description.
BIT_DEPTH_PYAV_ONLY: Final = 16
#: §21.2: never a lossy codec.
LOSSLESS_CODECS: Final = frozenset({"ffv1"})


@unique
class DisparityOutput(StrEnum):
    """What a `NeroArmsDisparitySource` emits per frame."""

    #: raw disparity as stored -- uint8 levels, or uint16 scaled by
    #: `2**subpixel_fractional_bits` (§21.16); needs no calibration
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
    subpixel_fractional_bits: int = 0
    """§21.16: stored values are `disparity * 2**this`. Defaults to 0 for the
    non-subpixel form, where the scale is 1; `_check_mode` refuses a subpixel
    declaration that leaves it unset, because then nothing can read the pixels."""
    bit_depth: Literal[8, 16] | None = None
    """§21.16: which store the frames are in, and therefore which decoder is
    legal -- 16-bit must go to PyAV, never torchcodec. Optional so a recording
    written before the field existed still loads; the container itself is the
    authority, and `check_bit_depth` refuses a disagreement rather than
    preferring one."""
    width: PositiveInt | None = None
    height: PositiveInt | None = None
    mono_sockets: str | None = None

    model_config = ConfigDict(extra="allow")

    @property
    def disparity_scale(self) -> int:
        """`2**subpixel_fractional_bits`; 1 when subpixel is off."""
        return 1 << self.subpixel_fractional_bits

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        check_disparity_mode(
            max_disparity=self.max_disparity,
            subpixel=self.subpixel,
            extended_disparity=self.extended_disparity,
            subpixel_fractional_bits=self.subpixel_fractional_bits,
            source="episode metadata",
        )

        return self

    def check_bit_depth(self, observed: int, *, source: str) -> None:
        """The declaration and the container must agree about the store."""
        check_store_against_mode(
            observed_bit_depth=observed,
            subpixel=self.subpixel,
            subpixel_fractional_bits=self.subpixel_fractional_bits,
            declared_bit_depth=self.bit_depth,
            source=source,
        )

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
            # `Literal[8, 16]` will not coerce a string, and it should not: it is
            # a routing key, so "16 in some other notation" is not a thing to be
            # lenient about. Int-ness is the only conversion made here.
            | (
                {"bit_depth": int(metadata["bit_depth"])}
                if "bit_depth" in metadata
                else {}
            )
        )

    def check_against(self, stereo: StereoCalibration, *, camera: str) -> None:
        """The per-episode declaration wins; a disagreement is a hard failure."""
        mismatched = {
            field: (declared, calibrated)
            for field in (
                "max_disparity",
                "subpixel",
                "extended_disparity",
                # §21.16: a scale mismatch is the quiet one. Both sides validate
                # internally, so 95<<3 against 95<<5 would only be caught here --
                # and it is a factor of 4 on every depth.
                "subpixel_fractional_bits",
            )
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


def check_store_against_mode(
    *,
    observed_bit_depth: int,
    subpixel: bool,
    subpixel_fractional_bits: int,
    declared_bit_depth: int | None = None,
    source: str,
) -> None:
    """§21.16: the store a stream is in must agree with the declared mode.

    A disagreement is not a preference to resolve, it is a corrupt recording: the
    declared `max_disparity` and scale describe values of one width while the file
    holds another, so every reading is off by a power of two and nothing in the
    pixels says so.

    Takes the fields rather than an object so the check reads identically off a
    `DisparityDeclaration` (which also carries `bit_depth`) and off a
    `StereoCalibration` (which does not) -- there must be exactly one rule here,
    not one per caller.
    """
    if declared_bit_depth is not None and declared_bit_depth != observed_bit_depth:
        logger.error(
            msg := "§21.16: declared `bit_depth` disagrees with the stream",
            source=source,
            declared=declared_bit_depth,
            observed=observed_bit_depth,
        )

        raise ValueError(msg)

    if (observed_bit_depth == BIT_DEPTH_PYAV_ONLY) != subpixel:
        # 8-bit under subpixel means the x2**bits values were truncated
        # somewhere. 16-bit without subpixel is the §21.12 stream -- measured on
        # the rig, subpixel reading back OFF while the device delivered uint16
        # with values to 6164 -- and its scale is whatever the device was
        # actually doing, i.e. unknown.
        logger.error(
            msg := "§21.16: the stream's store contradicts the declared "
            "`subpixel` mode, so the scale of its values is unknown",
            source=source,
            observed_bit_depth=observed_bit_depth,
            subpixel=subpixel,
            subpixel_fractional_bits=subpixel_fractional_bits,
        )

        raise ValueError(msg)


def disparity_to_depth(
    disparity: Tensor, stereo: StereoCalibration
) -> tuple[Tensor, Tensor]:
    """§21.4/§21.16: `(depth_m, valid)` from stored disparity.

    `depth_m = fx_mono_px * baseline_m / (raw / 2**subpixel_fractional_bits)`.
    The divisor comes from `stereo.disparity_scale`, i.e. from the **declared**
    mode -- never from the dtype or from a constant. A stream stored at 5
    fractional bits read as if it were 3 is wrong by exactly 4x everywhere, and
    the depth map still looks entirely plausible.

    Algebraically the scale is a multiplier on the numerator, which is how it is
    applied here: it keeps the clamp operating on raw stored units, so the
    invalid test and the guard against division blow-up stay exact integers.

    `disparity == 0` means **invalid, not zero distance**. The denominator is
    clamped *before* the division, so an invalid pixel can never produce `inf`
    (which would turn into `nan` the moment anything multiplies it by the mask)
    and never becomes a 0 m reading either -- it is 0.0 *and* masked out, and
    consumers must use the mask.
    """
    valid = disparity > 0
    safe = disparity.to(torch.float32).clamp(min=1.0)
    numerator = stereo.fx_mono * stereo.baseline_m * stereo.disparity_scale
    depth = torch.where(valid, numerator / safe, torch.zeros((), dtype=safe.dtype))

    return depth, valid
