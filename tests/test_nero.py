"""Tests for the nero-arms ingestion (data contract v0.1).

The unit tests below are self-contained. The end-to-end schema test needs the
recording share and is skipped when it is not mounted -- the dummy recordings
are ~2.4 GB and are deliberately not vendored into `tests/data`.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from rbyte.io.nero import (
    CAMERA_COND_DIM,
    CAMERAS,
    IMU_DIM,
    SIDES,
    STATE_DIM_9D,
    STATE_DIM_QUAT,
    STATUS_SENSORS,
    NeroArmsCalibration,
    NeroArmsDataFrameBuilder,
    canonicalize_quat,
    pose_9d_to_quat,
    pose_quat_to_9d,
    quat_slerp,
    quat_to_rot6d,
    rot6d_to_quat,
    state_9d_to_quat,
    state_quat_to_9d,
)
from rbyte.io.nero.rotation import matrix_to_quat, quat_to_matrix

DATA_DIR = Path(__file__).resolve().parent / "data" / "nero_arms"
CALIBRATION_PATH = DATA_DIR / "calibration.yaml"

#: the recording session root; override with `NERO_ARMS_DATA_DIR`
SESSION_DIR = Path(
    os.environ.get(
        "NERO_ARMS_DATA_DIR", "/nasa/drives/nero-arms/gloves-mcap-recordings/2026-07-17"
    )
)

#: §3.4: one glove period at 83.4 Hz
GLOVE_PERIOD_MS = 12.0
#: §2.5 status-flag channels per side actually present in the recordings
N_STATUS_SENSORS = 8
#: half a glove period -- the worst possible distance to a bracketing sample
GLOVE_HALF_PERIOD_MS = GLOVE_PERIOD_MS / 2


def _random_quats(n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))

    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# --------------------------------------------------------------- §5 rotations


def test_canonicalize_quat_flips_negative_qw() -> None:
    q = _random_quats(256)
    canonical = canonicalize_quat(q)

    assert (canonical[:, 3] >= 0.0).all(), "qw must be non-negative after §5.1"
    # q and -q are the same rotation
    assert np.allclose(np.abs(np.sum(canonical * q, axis=-1)), 1.0)
    # idempotent
    assert np.array_equal(canonicalize_quat(canonical), canonical)


def test_canonicalize_quat_preserves_rotation() -> None:
    q = _random_quats(64, seed=1)

    assert np.allclose(quat_to_matrix(q), quat_to_matrix(canonicalize_quat(q)))


def test_quat_rot6d_roundtrip() -> None:
    q = canonicalize_quat(_random_quats(256, seed=2))
    r6 = quat_to_rot6d(q)

    assert r6.shape == (256, 6)
    assert np.allclose(rot6d_to_quat(r6), q, atol=1e-9)


def test_rot6d_is_first_two_columns_of_r() -> None:
    q = canonicalize_quat(_random_quats(16, seed=3))
    m = quat_to_matrix(q)

    assert np.allclose(quat_to_rot6d(q), np.concatenate([m[:, :, 0], m[:, :, 1]], -1))


def test_identity_rotation_6d() -> None:
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    assert np.allclose(quat_to_rot6d(identity), [1, 0, 0, 0, 1, 0])


def test_pose_and_state_roundtrip() -> None:
    rng = np.random.default_rng(4)
    pose = np.concatenate(
        [rng.standard_normal((32, 3)), canonicalize_quat(_random_quats(32, seed=5))], -1
    )

    assert pose_quat_to_9d(pose).shape == (32, 9)
    assert np.allclose(pose_9d_to_quat(pose_quat_to_9d(pose)), pose, atol=1e-9)

    state = np.concatenate(
        [
            np.concatenate(
                [
                    rng.standard_normal((8, 3)),
                    canonicalize_quat(_random_quats(8, seed=6 + i)),
                ],
                -1,
            )
            for i in range(6)
        ]
        + [canonicalize_quat(_random_quats(8, seed=99))],
        axis=-1,
    )

    assert state.shape == (8, STATE_DIM_QUAT)
    assert state_quat_to_9d(state).shape == (8, STATE_DIM_9D)
    assert np.allclose(state_9d_to_quat(state_quat_to_9d(state)), state, atol=1e-9)


def test_matrix_quat_roundtrip() -> None:
    q = canonicalize_quat(_random_quats(256, seed=7))

    assert np.allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-9)


# ------------------------------------------------------------------ §3 slerp


def test_quat_slerp_endpoints_and_midpoint() -> None:
    q0 = canonicalize_quat(_random_quats(64, seed=8))
    q1 = canonicalize_quat(_random_quats(64, seed=9))

    assert np.allclose(np.abs(np.sum(quat_slerp(q0, q1, np.zeros(64)) * q0, -1)), 1.0)
    assert np.allclose(np.abs(np.sum(quat_slerp(q0, q1, np.ones(64)) * q1, -1)), 1.0)

    mid = quat_slerp(q0, q1, np.full(64, 0.5))

    assert np.allclose(np.linalg.norm(mid, axis=-1), 1.0), "slerp must stay on S3"
    # equidistant from both endpoints
    angle_0 = np.arccos(np.clip(np.abs(np.sum(mid * q0, -1)), -1, 1))
    angle_1 = np.arccos(np.clip(np.abs(np.sum(mid * q1, -1)), -1, 1))

    assert np.allclose(angle_0, angle_1, atol=1e-9)


def test_quat_slerp_takes_the_shortest_arc() -> None:
    q0 = canonicalize_quat(_random_quats(64, seed=10))
    q1 = canonicalize_quat(_random_quats(64, seed=11))
    t = np.random.default_rng(12).uniform(size=64)

    # slerping to -q1 must give the same rotation as slerping to q1
    assert np.allclose(
        quat_to_matrix(quat_slerp(q0, q1, t)), quat_to_matrix(quat_slerp(q0, -q1, t))
    )


def test_quat_slerp_is_not_nearest_neighbour() -> None:
    """A 90 degree rotation interpolated at t=0.5 must land at 45 degrees."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    mid = quat_slerp(q0, q1, np.array(0.5))

    assert np.allclose(mid, [0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)])


def test_quat_slerp_handles_identical_endpoints() -> None:
    q = canonicalize_quat(_random_quats(16, seed=13))
    out = quat_slerp(q, q, np.full(16, 0.3))

    assert np.isfinite(out).all()
    assert np.allclose(out, q)


# ------------------------------------------------------------ §7 calibration


def test_calibration_placeholder() -> None:
    calibration = NeroArmsCalibration.from_path(CALIBRATION_PATH)

    assert calibration.placeholder, "shipped calibration must be flagged placeholder"
    assert set(calibration.cameras) == set(CAMERAS)
    # §7.2: the cameras are heterogeneous
    assert calibration.cameras["base"].image_size == (1920, 1080)
    assert calibration.cameras["side_left"].image_size == (1280, 800)
    assert calibration.cameras["side_right"].image_size == (1280, 800)


def test_camera_cond_shape_and_identity_extrinsics() -> None:
    cond = NeroArmsCalibration.from_path(CALIBRATION_PATH).cond(CAMERAS)

    assert cond.shape == (len(CAMERAS), CAMERA_COND_DIM)
    assert cond.dtype == np.float32

    for row in cond:
        # placeholder intrinsics are zero, and normalising by W/H keeps them zero
        assert np.allclose(row[:4], 0.0)
        assert np.allclose(row[4:7], 0.0), "placeholder translation is the origin"
        # identity rotation as 6D -- not all-zero, so assert on it exactly
        assert np.allclose(row[7:], [1, 0, 0, 0, 1, 0])


def test_camera_cond_is_resolution_normalised() -> None:
    """Isotropic resize must leave the conditioning vector unchanged (§7.1)."""
    calibration = NeroArmsCalibration.from_path(CALIBRATION_PATH)
    camera = calibration.cameras["base"].model_copy(deep=True)
    camera.intrinsics.fx = 960.0
    camera.intrinsics.fy = 960.0
    camera.intrinsics.cx = 960.0
    camera.intrinsics.cy = 540.0
    cond = camera.cond

    scaled = camera.model_copy(deep=True)
    scaled.image_size = (480, 270)
    for field in ("fx", "fy", "cx", "cy"):
        setattr(scaled.intrinsics, field, getattr(scaled.intrinsics, field) * 0.25)

    assert np.allclose(cond, scaled.cond)


# ------------------------------------------------------- §6 schema / topology


def test_state_dims_match_the_contract() -> None:
    # §5.2 storage form: 6 poses x 7 + hub orientation quaternion
    assert STATE_DIM_QUAT == 6 * 7 + 4
    # §6.1 model-facing form: arm 9 + fingers 45 + hub rotation 6
    assert STATE_DIM_9D == 9 + 45 + 6
    assert IMU_DIM == 7 * 6
    assert SIDES == ("left", "right"), "index 0 is left, index 1 is right"
    # NOTE: §8 quotes 7 status-flag channels; the recordings carry 8 (§2.5 says
    # do not drop, so all 8 are carried). See the summary in the PR description.
    assert len(STATUS_SENSORS) == N_STATUS_SENSORS


def test_builder_rejects_unknown_reference_camera() -> None:
    with pytest.raises(ValueError, match="reference_camera"):
        NeroArmsDataFrameBuilder(cameras=["base"], reference_camera="side_left")


# ---------------------------------------------------------------- end to end

_EPISODES = (
    sorted(p.name for p in SESSION_DIR.iterdir() if (p / "data.mcap").is_file())
    if SESSION_DIR.is_dir()
    else []
)

requires_session = pytest.mark.skipif(
    not _EPISODES, reason=f"nero-arms recording session not mounted at {SESSION_DIR}"
)


@pytest.fixture(scope="session")
def nero_arms_dataframe() -> pl.DataFrame:
    builder = NeroArmsDataFrameBuilder(include_imu=True, include_status_flags=True)

    return builder(
        SESSION_DIR / _EPISODES[0] / "data.mcap", SESSION_DIR / "calibration.yaml"
    )


@requires_session
def test_dataframe_schema(nero_arms_dataframe: pl.DataFrame) -> None:
    c = SimpleNamespace(
        S=len(SIDES), D=STATE_DIM_QUAT, H=6, C=len(CAMERAS), K=CAMERA_COND_DIM
    )

    assert nero_arms_dataframe.schema == pl.Schema({
        "frame_index.base": pl.Int32,
        "frame_index.side_left": pl.Int32,
        "frame_index.side_right": pl.Int32,
        "state.pose": pl.Array(pl.Float32, (c.S, c.D)),
        "state.pose_rel_start": pl.Array(pl.Float32, (c.S, c.D)),
        "action.future_state": pl.Array(pl.Float32, (c.H, c.S, c.D)),
        "side_valid": pl.Array(pl.Boolean, (c.S,)),
        "align_residual_ms": pl.Float32,
        "camera_cond": pl.Array(pl.Float32, (c.C, c.K)),
        "camera_cond.placeholder": pl.Boolean,
        "goal.xyz": pl.Array(pl.Float32, (c.S, 3)),
        "goal.frame_index.base": pl.Int32,
        "goal.frame_index.side_left": pl.Int32,
        "goal.frame_index.side_right": pl.Int32,
        "state.imu": pl.Array(pl.Float32, (c.S, IMU_DIM)),
        "state.status_flags": pl.Array(pl.Int32, (c.S, len(STATUS_SENSORS))),
    })


@requires_session
def test_alignment_residual_within_one_glove_period(
    nero_arms_dataframe: pl.DataFrame,
) -> None:
    residual = nero_arms_dataframe["align_residual_ms"]

    assert residual.null_count() == 0, "no extrapolation: rows must be dropped instead"
    assert float(residual.min()) >= 0.0
    # §3.4: fail loudly above one glove period. Measured over all 104 episodes
    # the worst case is 6.47 ms and the mean is 3.00 ms.
    assert float(residual.max()) < GLOVE_PERIOD_MS, (
        f"alignment residual {residual.max()} ms exceeds one glove period"
    )
    assert float(residual.mean() or 0.0) < GLOVE_HALF_PERIOD_MS, (
        "mean residual should sit well inside half a glove period"
    )


@requires_session
def test_quaternions_are_canonical(nero_arms_dataframe: pl.DataFrame) -> None:
    state = nero_arms_dataframe["state.pose"].to_numpy()
    right = state[:, SIDES.index("right")]
    # 6 poses of 7, then a bare quaternion
    qw = np.concatenate([right[:, 6:42:7], right[:, -1:]], axis=-1)

    assert (qw >= 0.0).all(), "§5.1: qw must be non-negative after ingestion"

    quats = np.concatenate(
        [right[:, :42].reshape(-1, 6, 7)[..., 3:], right[:, None, 42:]], axis=1
    )

    assert np.allclose(np.linalg.norm(quats, axis=-1), 1.0, atol=1e-5)


@requires_session
def test_missing_side_is_zeros_plus_mask(nero_arms_dataframe: pl.DataFrame) -> None:
    left, right = SIDES.index("left"), SIDES.index("right")
    side_valid = nero_arms_dataframe["side_valid"].to_numpy()

    assert (side_valid == [False, True]).all(), (
        "the dummy is right-hand only; left must be masked out, not dropped"
    )

    for column in ("state.pose", "state.pose_rel_start", "state.imu", "goal.xyz"):
        values = nero_arms_dataframe[column].to_numpy()

        assert (values[:, left] == 0).all(), f"{column}: masked side must be zeros"
        assert values[:, right].any(), f"{column}: valid side must not be all-zero"

    assert (nero_arms_dataframe["state.status_flags"].to_numpy()[:, left] == 0).all()


@requires_session
def test_pose_rel_start_is_zero_at_the_first_frame(
    nero_arms_dataframe: pl.DataFrame,
) -> None:
    right = SIDES.index("right")
    first = nero_arms_dataframe["state.pose_rel_start"].to_numpy()[0, right]

    assert np.allclose(first[:3], 0.0, atol=1e-6), "arm translation is start-relative"
    assert np.allclose(first[3:7], [0, 0, 0, 1], atol=1e-6), "arm rotation is identity"
    assert np.allclose(first[-4:], [0, 0, 0, 1], atol=1e-6), "hub rotation is identity"
    # fingers are already hub-relative and must be left untouched
    assert np.allclose(
        first[7:42], nero_arms_dataframe["state.pose"].to_numpy()[0, right, 7:42]
    )


@requires_session
def test_action_is_the_future_state_chunk(nero_arms_dataframe: pl.DataFrame) -> None:
    state = nero_arms_dataframe["state.pose"].to_numpy()
    future = nero_arms_dataframe["action.future_state"].to_numpy()
    horizon = future.shape[1]

    assert len(state) > horizon

    for h in range(horizon):
        # action(t)[h] == state(t + 1 + h); the last H rows are dropped, so the
        # comparison is exact for every emitted row
        assert np.array_equal(
            future[: len(state) - horizon, h], state[1 + h :][: len(state) - horizon]
        ), f"action.future_state[:, {h}] is not state[t + {1 + h}]"


@requires_session
def test_cameras_are_joined_on_their_own_frame_index(
    nero_arms_dataframe: pl.DataFrame,
) -> None:
    df = nero_arms_dataframe

    # the reference camera indexes the grid and must be strictly increasing
    assert df["frame_index.base"].is_sorted(descending=False)
    assert int(df["frame_index.base"].diff().drop_nulls().min()) >= 1  # ty: ignore[invalid-argument-type]

    # the sides are matched independently and may legitimately disagree with the
    # reference index -- 200 vs 201 frames, §3
    for camera in ("side_left", "side_right"):
        assert int(df[f"frame_index.{camera}"].min()) >= 0  # ty: ignore[invalid-argument-type]
        assert df[f"frame_index.{camera}"].is_sorted(descending=False)


@requires_session
def test_goal_is_the_final_frame_and_final_arm_position(
    nero_arms_dataframe: pl.DataFrame,
) -> None:
    df = nero_arms_dataframe

    for camera in CAMERAS:
        goal = df[f"goal.frame_index.{camera}"]

        assert goal.n_unique() == 1, "§9: goal is constant within an episode"
        assert goal[0] >= df[f"frame_index.{camera}"].max()

    goal_xyz = df["goal.xyz"].to_numpy()

    assert (goal_xyz == goal_xyz[0]).all(), "§9: goal is constant within an episode"


@requires_session
def test_calibration_placeholder_is_visible_per_sample(
    nero_arms_dataframe: pl.DataFrame,
) -> None:
    placeholder = nero_arms_dataframe["camera_cond.placeholder"]

    assert placeholder.all(), (
        "consumers must be able to assert on `placeholder` from a batch (§7)"
    )


@requires_session
def test_episode_lengths_are_not_uniform() -> None:
    """§1: episode length varies; nothing may assume a fixed length."""
    builder = NeroArmsDataFrameBuilder()
    lengths = {
        episode: len(
            builder(
                SESSION_DIR / episode / "data.mcap", SESSION_DIR / "calibration.yaml"
            )
        )
        for episode in _EPISODES[:8]
    }

    assert len(set(lengths.values())) > 1, f"expected varying lengths, got {lengths}"
