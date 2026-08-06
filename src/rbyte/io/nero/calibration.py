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
from typing import Annotated, Self, final

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field, FilePath, validate_call
from structlog import get_logger

from rbyte.io.nero.rotation import matrix_to_quat, quat_to_rot6d

__all__ = ["CAMERA_COND_DIM", "CameraModel", "NeroArmsCalibration"]

logger = get_logger(__name__)

#: §7.1: `fx/W, fy/H, cx/W, cy/H` + `t_world_cam` (3) + `R_world_cam` as 6D.
CAMERA_COND_DIM = 13


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


@final
class NeroArmsCalibration(BaseModel):
    version: int
    world_frame: str
    cameras: dict[str, CameraCalibration]
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

    def cond(self, cameras: Sequence[str]) -> npt.NDArray[np.float32]:
        """§7.1 conditioning matrix, `(len(cameras), 13)` float32."""
        if missing := set(cameras) - self.cameras.keys():
            logger.error(msg := "calibration missing cameras", cameras=sorted(missing))

            raise ValueError(msg)

        return np.stack([self.cameras[camera].cond for camera in cameras])
