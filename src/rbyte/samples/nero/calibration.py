"""Camera calibration sidecar for nero-arms (data contract §7).

Calibration is not present in the MCAP. It is read from a YAML sidecar keyed by
camera name, resolved per recording-session directory, so that real calibration
is a file drop. Until then the file carries `placeholder: true` and every
consumer is expected to assert on it -- `NeroArmsCalibration.placeholder` is
also emitted as a per-sample column so the assertion can be made on a batch.
"""

from collections.abc import Sequence
from enum import StrEnum, auto, unique
from pathlib import Path
from typing import Annotated, Final, Self, final

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    model_validator,
    validate_call,
)
from structlog import get_logger

from rbyte.samples.nero.rotation import matrix_to_quat, quat_to_rot6d

__all__ = [
    "CAMERA_COND_DIM",
    "MAX_DISPARITY",
    "SUBPIXEL_FRACTIONAL_BITS",
    "CameraModel",
    "NeroArmsCalibration",
    "StereoCalibration",
]

logger = get_logger(__name__)

#: §7.1: `fx/W, fy/H, cx/W, cy/H` + `t_world_cam` (3) + `R_world_cam` as 6D.
CAMERA_COND_DIM = 13

#: §21.1, keyed by `extended_disparity`: the INTEGER disparity levels the matcher
#: produces, before any subpixel scaling. `subpixel` multiplies these by
#: `2**subpixel_fractional_bits` and the product is what the stream stores.
MAX_DISPARITY: Final[dict[bool, int]] = {False: 95, True: 191}

#: §21.16: depthai's subpixel modes. A fresh `StereoDepthConfig` reports 5
#: fractional bits (x32) and a preset-built node was measured at 3 (x8), so the
#: exponent is read from the recorder's declaration and NEVER assumed -- getting
#: it wrong scales every depth by 4x with nothing in the frames to reveal it.
SUBPIXEL_FRACTIONAL_BITS: Final[frozenset[int]] = frozenset({3, 4, 5})

#: The widest store a disparity value may occupy. 191 << 5 == 6112, so every
#: permitted mode fits comfortably.
MAX_STORED_DISPARITY: Final = 65535


def check_disparity_mode(
    *,
    max_disparity: int,
    subpixel: bool,
    extended_disparity: bool,
    subpixel_fractional_bits: int = 0,
    source: str,
) -> None:
    """§21.16: the declared stereo mode must be internally consistent.

    **`subpixel` is no longer refused.** It is the primary form now: §21.14
    measured that integer disparity cannot resolve the manipulated object at the
    rig's working distance -- a 2 cm cube at 0.64 m spans 1.1 disparity levels,
    one quantisation step -- while at 5 fractional bits the same delta spans ~35
    units. What used to be a veto becomes a *scale* that has to be declared.

    So what is refused is an UNINTERPRETABLE declaration:

    * `subpixel` on with fractional bits outside `SUBPIXEL_FRACTIONAL_BITS`. The
      stored values are `disparity * 2**bits` and the metric conversion needs the
      exponent; a default would be silently wrong by a power of two.
    * `subpixel` off with a non-zero scale, which is a contradiction.
    * a `max_disparity` that does not match `MAX_DISPARITY[extended] << bits`.
      It is *declared*, never inferred (§17.1), but a declaration that
      contradicts the mode is a recorder bug -- and assuming 95 when 191 was
      recorded scales every depth by 2x.
    """
    if subpixel and subpixel_fractional_bits not in SUBPIXEL_FRACTIONAL_BITS:
        logger.error(
            msg := "§21.16: `subpixel` is on but `subpixel_fractional_bits` is not "
            "a known scale -- stored values are disparity * 2**bits and without "
            "the exponent they cannot be converted to metres",
            source=source,
            subpixel_fractional_bits=subpixel_fractional_bits,
            expected=sorted(SUBPIXEL_FRACTIONAL_BITS),
        )

        raise ValueError(msg)

    if not subpixel and subpixel_fractional_bits:
        logger.error(
            msg := "§21.16: `subpixel` is off but `subpixel_fractional_bits` is "
            "non-zero -- the scale of the stored values is contradictory",
            source=source,
            subpixel_fractional_bits=subpixel_fractional_bits,
        )

        raise ValueError(msg)

    expected = MAX_DISPARITY[extended_disparity] << subpixel_fractional_bits
    if max_disparity != expected:
        logger.error(
            msg := "§21.1: declared `max_disparity` contradicts `extended_disparity`",
            source=source,
            max_disparity=max_disparity,
            extended_disparity=extended_disparity,
            subpixel_fractional_bits=subpixel_fractional_bits,
            expected=expected,
        )

        raise ValueError(msg)

    if max_disparity > MAX_STORED_DISPARITY:
        logger.error(
            msg := "§21.16: declared `max_disparity` does not fit a 16-bit store",
            source=source,
            max_disparity=max_disparity,
        )

        raise ValueError(msg)


@unique
class CameraModel(StrEnum):
    pinhole = auto()
    opencv = auto()
    opencv_fisheye = auto()


class Intrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float

    model_config = ConfigDict(extra="forbid")


class CameraCalibration(BaseModel):
    model: CameraModel
    image_size: tuple[int, int]
    intrinsics: Intrinsics
    distortion: Sequence[float] = ()
    T_world_cam: Annotated[Sequence[Sequence[float]], Field(min_length=4, max_length=4)]

    # Provenance, written by read_oak_intrinsics.py when the values come from
    # device EEPROM. Optional so hand-written and placeholder files still load.
    mxid: str | None = None
    socket: str | None = None
    rotated_180: bool = False
    """The recorder rotates every camera 180 degrees, and the EEPROM intrinsics
    describe the unrotated sensor, so `intrinsics`/`distortion` here must already
    carry that correction. This flag records that it was applied -- it is a
    marker, not an instruction: nothing downstream re-applies it."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @property
    def cond(self) -> npt.NDArray[np.float32]:
        """§7.1 conditioning vector, `(13,)` float32.

        The intrinsics are resolution-normalised, which makes the vector
        invariant to an *isotropic* resize of the image. Any anisotropic resize,
        crop or letterbox applied to the pixels must be propagated into
        `intrinsics`/`image_size` here as well (§7.2), otherwise the
        conditioning vector describes a camera that does not match the pixels.
        """
        w, h = self.image_size
        t_world_cam = np.asarray(self.T_world_cam, dtype=np.float64)
        r_world_cam = t_world_cam[:3, :3]

        return np.concatenate([
            [
                self.intrinsics.fx / w,
                self.intrinsics.fy / h,
                self.intrinsics.cx / w,
                self.intrinsics.cy / h,
            ],
            t_world_cam[:3, 3],
            quat_to_rot6d(matrix_to_quat(r_world_cam)),
        ]).astype(np.float32)


class StereoCalibration(BaseModel):
    """§21: the **mono pair** behind a disparity stream.

    `fx_mono` is the mono (`CAM_B`/`CAM_C`) focal length, **not** the `fx` under
    `cameras:` -- that one is `CAM_A`, the RGB camera, a different and much
    narrower lens (§21.5). Using it would scale every depth wrong, and did:
    an earlier MinZ of ~0.70 m came from plugging the RGB `fx` into the formula.

    `image_size` is the resolution `fx_mono` is expressed at, and the disparity
    frames MUST be decoded at exactly that size: a resize scales `fx` but *not*
    the stored disparity values, so `fx * baseline / disparity` would come out
    wrong by the resize factor. The depth stream is therefore never resized.
    """

    fx_mono: PositiveFloat
    """px, at `image_size`. `c.getCameraIntrinsics(CAM_B, resizeWidth=W, ...)`."""
    baseline_m: PositiveFloat
    """m. `c.getBaselineDistance()` returns cm."""
    image_size: tuple[PositiveInt, PositiveInt]
    """`(width, height)` of the disparity frames."""

    # §21.3 / §17.1: declared, never inferred.
    max_disparity: PositiveInt
    """The largest STORED value, i.e. already scaled by `disparity_scale`."""
    subpixel: bool = False
    extended_disparity: bool = False
    subpixel_fractional_bits: int = 0
    """§21.16: stored values are `disparity * 2**this`, so the metric conversion
    divides by it first. 0 when `subpixel` is off. Declared by the recorder and
    never assumed -- depthai reports 5 on a fresh config and was measured at 3 on
    a preset-built node, a 4x error with nothing in the frames to reveal it."""
    mono_sockets: tuple[str, str] | None = None

    placeholder: bool = True
    """A guessed block is worse than none: it scales every depth silently, so
    `NeroArmsCalibration.stereo_for` refuses to use one. The `base` OAK-D W
    values are MEASURED off the device; a second rig starts out placeholder."""

    model_config = ConfigDict(extra="forbid")

    @property
    def disparity_scale(self) -> int:
        """`2**subpixel_fractional_bits` -- the divisor the stored values carry.

        1 when `subpixel` is off, so the plain `fx * baseline / disparity` form
        is the special case rather than a separate code path.
        """
        return 1 << self.subpixel_fractional_bits

    @property
    def min_depth_m(self) -> float:
        """Closest resolvable distance -- depth at `max_disparity`.

        Measured `base` rig (`fx_mono` 576.37 px @ 1280x800, baseline 75.00 mm):
        **0.455 m** in the default mode, **0.226 m** with extended disparity.
        `CAMERA_RIG.md` §4.2: check this against the actual mount height, since
        an overhead camera sitting inside its own minimum range is unusable.

        Unchanged by subpixel: `max_disparity` and the scale move together, so
        the closest resolvable distance is a property of the optics, not of the
        quantisation. What subpixel buys is resolution *between* the levels.
        """
        return (
            self.fx_mono * self.baseline_m * self.disparity_scale / self.max_disparity
        )

    @property
    def max_depth_m(self) -> float:
        """Depth at the coarsest non-invalid stored value (43.2 m at scale 1).

        With subpixel that value is one *fraction* of a level rather than a whole
        one, so the far limit moves out by `disparity_scale` -- which is the same
        statement as "the quantisation got finer", read at the other end.
        """
        return self.fx_mono * self.baseline_m * self.disparity_scale

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        check_disparity_mode(
            max_disparity=self.max_disparity,
            subpixel=self.subpixel,
            extended_disparity=self.extended_disparity,
            subpixel_fractional_bits=self.subpixel_fractional_bits,
            source="calibration.stereo",
        )

        return self


@final
class NeroArmsCalibration(BaseModel):
    version: int
    world_frame: str
    cameras: dict[str, CameraCalibration]
    stereo: dict[str, StereoCalibration] = {}
    """§21, keyed by camera name like `cameras`. Carries the MONO pair, which
    `cameras` does not; raw disparity ingests fine without it."""
    placeholder: bool = True

    intrinsics_source: str | None = None
    """Where the intrinsics came from -- "eeprom" when read off the devices by
    read_oak_intrinsics.py, None for hand-written or placeholder files."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    @validate_call
    def from_path(cls, path: FilePath) -> Self:
        with Path(path).open(encoding="utf-8") as f:
            calibration = cls.model_validate(yaml.safe_load(f))

        if calibration.placeholder:
            logger.warning(
                "using PLACEHOLDER camera calibration -- extrinsics are identity "
                "and intrinsics are zero; `placeholder` must be false before any "
                "real training run",
                path=Path(path).as_posix(),
            )

        return calibration

    def stereo_for(self, camera: str) -> StereoCalibration:
        """§21.4: the mono pair for `camera`, or a loud failure.

        Emitting raw disparity does not need this; metric depth does, and there
        is no safe default -- a missing or placeholder block would silently scale
        every depth reading, so it is refused.
        """
        match self.stereo.get(camera):
            case None:
                logger.error(
                    msg := "§21.4: no `stereo` calibration for camera -- metric "
                    "depth needs the MONO pair's `fx_mono` and `baseline_m`, "
                    "which are NOT the `cameras` intrinsics (those are the RGB "
                    "CAM_A). Measure them, or ingest raw disparity instead.",
                    camera=camera,
                    available=sorted(self.stereo),
                )

                raise ValueError(msg)

            case stereo if stereo.placeholder:
                logger.error(
                    msg := "§21.4: `stereo` calibration is still PLACEHOLDER -- "
                    "metric depth would be silently wrong by whatever the "
                    "guessed `fx_mono`/`baseline_m` are off by",
                    camera=camera,
                    fx_mono=stereo.fx_mono,
                    baseline_m=stereo.baseline_m,
                )

                raise ValueError(msg)

            case stereo:
                return stereo

    def cond(self, cameras: Sequence[str]) -> npt.NDArray[np.float32]:
        """§7.1 conditioning matrix, `(len(cameras), 13)` float32."""
        if missing := set(cameras) - self.cameras.keys():
            logger.error(msg := "calibration missing cameras", cameras=sorted(missing))

            raise ValueError(msg)

        return np.stack([self.cameras[camera].cond for camera in cameras])
