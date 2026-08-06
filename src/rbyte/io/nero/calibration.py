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

from rbyte.io.nero.rotation import matrix_to_quat, quat_to_rot6d

__all__ = [
    "CAMERA_COND_DIM",
    "MAX_DISPARITY",
    "CameraModel",
    "NeroArmsCalibration",
    "StereoCalibration",
]

logger = get_logger(__name__)

#: §7.1: `fx/W, fy/H, cx/W, cy/H` + `t_world_cam` (3) + `R_world_cam` as 6D.
CAMERA_COND_DIM = 13

#: §21.1, keyed by `extended_disparity`. `subpixel` would make it 760, which does
#: not fit in the 8 bits the stream is stored in -- hence the hard assertion.
MAX_DISPARITY: Final[dict[bool, int]] = {False: 95, True: 191}


def check_disparity_mode(
    *, max_disparity: int, subpixel: bool, extended_disparity: bool, source: str
) -> None:
    """§21.1/§21.3: the declared stereo mode must be 8-bit representable.

    `subpixel` multiplies disparity by 8 and needs 16 bits; reading such a stream
    as uint8 is silently wrong by a factor of 8, so it is refused rather than
    warned about. `max_disparity` is *declared*, never inferred (§17.1) -- but a
    declaration that contradicts `extended_disparity` is a recorder bug, and
    assuming 95 when 191 was recorded scales every depth by 2x.
    """
    if subpixel:
        logger.error(
            msg := "§21.1: `subpixel` must be false -- subpixel disparity needs "
            "16 bits and an 8-bit read is silently wrong by a factor of 8",
            source=source,
        )

        raise ValueError(msg)

    if max_disparity != (expected := MAX_DISPARITY[extended_disparity]):
        logger.error(
            msg := "§21.1: declared `max_disparity` contradicts `extended_disparity`",
            source=source,
            max_disparity=max_disparity,
            extended_disparity=extended_disparity,
            expected=expected,
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
    subpixel: bool = False
    extended_disparity: bool = False
    mono_sockets: tuple[str, str] | None = None

    placeholder: bool = True
    """A guessed block is worse than none: it scales every depth silently, so
    `NeroArmsCalibration.stereo_for` refuses to use one. The `base` OAK-D W
    values are MEASURED off the device; a second rig starts out placeholder."""

    model_config = ConfigDict(extra="forbid")

    @property
    def min_depth_m(self) -> float:
        """Closest resolvable distance -- depth at `max_disparity`.

        Measured `base` rig (`fx_mono` 576.37 px @ 1280x800, baseline 75.00 mm):
        **0.455 m** in the default mode, **0.226 m** with extended disparity.
        `CAMERA_RIG.md` §4.2: check this against the actual mount height, since
        an overhead camera sitting inside its own minimum range is unusable.
        """
        return self.fx_mono * self.baseline_m / self.max_disparity

    @property
    def max_depth_m(self) -> float:
        """Depth at disparity 1, the coarsest non-invalid level (43.2 m here)."""
        return self.fx_mono * self.baseline_m

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        check_disparity_mode(
            max_disparity=self.max_disparity,
            subpixel=self.subpixel,
            extended_disparity=self.extended_disparity,
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
