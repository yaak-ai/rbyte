"""Layout constants for the nero-arms state/action vectors (data contract §6).

Storage form (what rbyte emits, §5.2) packs, per side:

| slice     | block                             | poses | dims |
|-----------|-----------------------------------|-------|------|
| `[0:7]`   | arm in the coil-pro world frame   | 1     | 7    |
| `[7:42]`  | finger tips relative to the hub   | 5     | 35   |
| `[42:46]` | hub orientation (rotation only)   | 1     | 4    |

giving `STATE_DIM_QUAT == 46` per side. `state_quat_to_9d` maps this onto the
model-facing 60-dim 9D form of §6.1 (arm 9 + fingers 45 + hub rotation 6).
"""

from typing import Final

import numpy as np
import numpy.typing as npt

from rbyte.io.nero.rotation import (
    pose_9d_to_quat,
    pose_quat_to_9d,
    quat_to_rot6d,
    rot6d_to_quat,
)

__all__ = [
    "CAMERAS",
    "FINGERS",
    "IMU_DIM",
    "IMU_SEGMENTS",
    "SIDES",
    "STATE_DIM_9D",
    "STATE_DIM_QUAT",
    "STATUS_SENSORS",
    "state_9d_to_quat",
    "state_quat_to_9d",
]

SIDES: Final = ("left", "right")
FINGERS: Final = ("thumb", "index", "middle", "ring", "little")
CAMERAS: Final = ("base", "side_left", "side_right")

#: §6.1 auxiliary block: 7 segments x (angular_velocity + proper_acceleration).
IMU_SEGMENTS: Final = (
    "hub",
    "arm",
    "thumb_finger",
    "index_finger",
    "middle_finger",
    "ring_finger",
    "little_finger",
)
IMU_DIM: Final = len(IMU_SEGMENTS) * 6

#: §2.5 auxiliary block. NOTE: the recordings carry **8** `status_flags` topics per
#: side, not the 7 quoted in the §8 schema table -- `arm_sensor.coil_pro` and
#: `arm_sensor` both publish one. §2.5 says "do not drop", so all 8 are carried.
STATUS_SENSORS: Final = (
    "arm_sensor.coil_pro",
    "arm_sensor",
    "hub",
    *(f"{finger}_finger_sensor" for finger in FINGERS),
)

_N_POSES: Final = 1 + len(FINGERS)
STATE_DIM_QUAT: Final = _N_POSES * 7 + 4
STATE_DIM_9D: Final = _N_POSES * 9 + 6


def state_quat_to_9d(state: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """`(..., 46)` storage state -> `(..., 60)` model-facing 9D state (§6.1)."""
    state = np.asarray(state, dtype=np.float64)
    poses = state[..., : _N_POSES * 7].reshape(*state.shape[:-1], _N_POSES, 7)
    hub = state[..., _N_POSES * 7 :]

    return np.concatenate(
        [
            pose_quat_to_9d(poses).reshape(*state.shape[:-1], _N_POSES * 9),
            quat_to_rot6d(hub),
        ],
        axis=-1,
    )


def state_9d_to_quat(state: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """`(..., 60)` model-facing 9D state -> `(..., 46)` storage state (§6.1)."""
    state = np.asarray(state, dtype=np.float64)
    poses = state[..., : _N_POSES * 9].reshape(*state.shape[:-1], _N_POSES, 9)
    hub = state[..., _N_POSES * 9 :]

    return np.concatenate(
        [
            pose_9d_to_quat(poses).reshape(*state.shape[:-1], _N_POSES * 7),
            rot6d_to_quat(hub),
        ],
        axis=-1,
    )
