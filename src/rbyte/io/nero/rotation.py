"""Rotation utilities for the nero-arms data contract (§5).

Quaternions are stored and manipulated in `(qx, qy, qz, qw)` order, matching the
`Transform` / `Orientation` protobuf field order in the source recordings.

The *storage* form is a canonicalised quaternion (7 floats per pose, §5.2). The
*model-facing* form is 3 translation + 6D continuous rotation (9 floats per pose,
§5.3); the conversion helpers live here so that ingestion and the rmind input
boundary share one implementation.
"""

from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "POSE_DIM_9D",
    "POSE_DIM_QUAT",
    "canonicalize_quat",
    "pose_9d_to_quat",
    "pose_quat_to_9d",
    "quat_slerp",
    "quat_to_rot6d",
    "rot6d_to_quat",
]

POSE_DIM_QUAT: Final = 7
POSE_DIM_9D: Final = 9

_EPS: Final = 1e-8


def canonicalize_quat(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Remove the quaternion double cover by flipping so that `qw >= 0` (§5.1)."""
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., 3:4] < 0.0, -q, q)


def quat_slerp(
    q0: npt.ArrayLike, q1: npt.ArrayLike, t: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Spherical linear interpolation along the shortest arc (§3.3).

    `t` broadcasts against the leading dimensions of `q0`/`q1`.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[..., None]

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    # shortest arc
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    colinear = sin_theta < _EPS
    sin_theta_safe = np.where(colinear, 1.0, sin_theta)

    w0 = np.where(colinear, 1.0 - t, np.sin((1.0 - t) * theta) / sin_theta_safe)
    w1 = np.where(colinear, t, np.sin(t * theta) / sin_theta_safe)

    q = w0 * q0 + w1 * q1
    norm = np.linalg.norm(q, axis=-1, keepdims=True)

    return q / np.where(norm < _EPS, 1.0, norm)


def quat_to_matrix(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """`(..., 4)` quaternion `(qx, qy, qz, qw)` -> `(..., 3, 3)` rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q / np.where(norm < _EPS, 1.0, norm), -1, 0)

    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(m: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """`(..., 3, 3)` rotation matrix -> canonicalised `(..., 4)` quaternion."""
    m = np.asarray(m, dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    # branchless Shepperd: build all four candidates, pick the numerically largest
    candidates = np.stack(
        [
            np.stack(
                [
                    1.0 + trace,
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ],
                axis=-1,
            ),
            np.stack(
                [
                    m[..., 2, 1] - m[..., 1, 2],
                    1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                    m[..., 0, 1] + m[..., 1, 0],
                    m[..., 0, 2] + m[..., 2, 0],
                ],
                axis=-1,
            ),
            np.stack(
                [
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 0, 1] + m[..., 1, 0],
                    1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                    m[..., 1, 2] + m[..., 2, 1],
                ],
                axis=-1,
            ),
            np.stack(
                [
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 1, 2] + m[..., 2, 1],
                    1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    # (..., 4 candidates, 4 components) in (w, x, y, z) order
    best = np.argmax(np.abs(candidates[..., 0]), axis=-1)
    wxyz = np.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    norm = np.linalg.norm(wxyz, axis=-1, keepdims=True)
    wxyz /= np.where(norm < _EPS, 1.0, norm)

    return canonicalize_quat(np.roll(wxyz, -1, axis=-1))


def quat_to_rot6d(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """`(..., 4)` quaternion -> `(..., 6)` continuous rotation (Zhou et al., §5.3).

    The 6D representation is the first two *columns* of the rotation matrix.
    """
    m = quat_to_matrix(q)

    return np.concatenate([m[..., 0], m[..., 1]], axis=-1)


def rot6d_to_quat(r: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """`(..., 6)` continuous rotation -> canonicalised `(..., 4)` quaternion."""
    r = np.asarray(r, dtype=np.float64)
    a, b = r[..., :3], r[..., 3:]

    norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
    e0 = a / np.where(norm_a < _EPS, 1.0, norm_a)
    # not in-place: `b` is a view onto the caller's array
    b = b - np.sum(e0 * b, axis=-1, keepdims=True) * e0  # noqa: PLR6104
    norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
    e1 = b / np.where(norm_b < _EPS, 1.0, norm_b)
    e2 = np.cross(e0, e1)

    return matrix_to_quat(np.stack([e0, e1, e2], axis=-1))


def pose_quat_to_9d(pose: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """`(..., 7)` `(x, y, z, qx, qy, qz, qw)` -> `(..., 9)` `(xyz, rot6d)` (§5.3)."""
    pose = np.asarray(pose, dtype=np.float64)

    return np.concatenate([pose[..., :3], quat_to_rot6d(pose[..., 3:])], axis=-1)


def pose_9d_to_quat(pose: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """`(..., 9)` `(xyz, rot6d)` -> `(..., 7)` `(x, y, z, qx, qy, qz, qw)` (§5.3)."""
    pose = np.asarray(pose, dtype=np.float64)

    return np.concatenate([pose[..., :3], rot6d_to_quat(pose[..., 3:])], axis=-1)
