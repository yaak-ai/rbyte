"""Tests for the nero-arms ingestion (data contract v0.1).

The unit tests below are self-contained. The end-to-end schema test needs the
recording share and is skipped when it is not mounted -- the dummy recordings
are ~2.4 GB and are deliberately not vendored into `tests/data`.
"""

import os
import subprocess  # noqa: S404
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import polars as pl
import pytest
import torch
import yaml
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.writer import Writer
from pydantic import ValidationError
from torchcodec.decoders import VideoDecoder

from rbyte.io.nero import (
    CAMERA_COND_DIM,
    CAMERAS,
    DISPARITY_METADATA_PREFIX,
    DISPARITY_TOPIC_PREFIX,
    FINGERS,
    IMU_DIM,
    MAX_DISPARITY,
    SIDES,
    STATE_DIM_9D,
    STATE_DIM_QUAT,
    STATUS_SENSORS,
    DisparityDeclaration,
    DisparityOutput,
    NeroArmsCalibration,
    NeroArmsDataFrameBuilder,
    NeroArmsDisparityFrameSource,
    StereoCalibration,
    canonicalize_quat,
    disparity_to_depth,
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
    assert float(residual.min()) >= 0.0  # ty: ignore[invalid-argument-type]
    # §3.4: fail loudly above one glove period. Measured over all 104 episodes
    # the worst case is 6.47 ms and the mean is 3.00 ms.
    assert float(residual.max()) < GLOVE_PERIOD_MS, (  # ty: ignore[invalid-argument-type]
        f"alignment residual {residual.max()} ms exceeds one glove period"
    )
    assert float(residual.mean() or 0.0) < GLOVE_HALF_PERIOD_MS, (  # ty: ignore[invalid-argument-type]
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


# ============================================================ §21 depth stream
#
# No recording carries depth yet (§21 is a design section), so everything below
# runs against a synthesised episode: FFV1-encoded uint8 disparity frames plus a
# minimal MCAP carrying the `observation.disparity.base` index topic and the
# §21.3 episode-metadata declaration.


#: the synthetic rig. `fx_mono * baseline_m` == 30.0 m*px, so disparity 1 is
#: 30 m and disparity 95 is 30/95 m -- both hand-checkable.
FX_MONO = 400.0
#: the MEASURED `base` OAK-D W mono pair (CAM_B @ 1280x800), §21.5.
FX_MONO_BASE = 576.37
#: derived range at the measured values -- `CAMERA_RIG.md` §4.2
MIN_DEPTH_M = 0.455
MIN_DEPTH_EXTENDED_M = 0.226
MAX_DEPTH_M = 43.2
BASELINE_M = 0.075
DISPARITY_SIZE = (64, 48)  # (width, height)
N_DISPARITY_FRAMES = 58
N_CAMERA_FRAMES = 60
#: §21.1: extended disparity halves MinZ and doubles the level count.
EXTENDED_MAX_DISPARITY = 191
#: §21.3: the disparity stream starts two camera frames late, so its own
#: `frame_index` is NOT the camera's -- which is the point of the separate join.
DISPARITY_START_FRAME = 2

_GLOVE_PERIOD_NS = 11_987_000
_CAMERA_PERIOD_NS = 33_333_333
_LATENCY_NS = 5_000_000
_T0_NS = 1_000_000_000
_GLOVE_EPOCH_NS = 700_000_000_000
_CAMERA_EPOCH_NS = 300_000_000_000


def _stereo(**overrides: object) -> StereoCalibration:
    return StereoCalibration.model_validate(
        {
            "fx_mono": FX_MONO,
            "baseline_m": BASELINE_M,
            "image_size": DISPARITY_SIZE,
            "max_disparity": MAX_DISPARITY[False],
            "subpixel": False,
            "extended_disparity": False,
            "placeholder": False,
        }
        | overrides
    )


def _declaration(**overrides: str) -> dict[str, str]:
    """§21.3 episode metadata as the recorder writes it: `dict[str, str]`."""
    width, height = DISPARITY_SIZE

    return {
        "max_disparity": str(MAX_DISPARITY[False]),
        "subpixel": "false",
        "extended_disparity": "false",
        "width": str(width),
        "height": str(height),
        "mono_sockets": "CAM_B,CAM_C",
    } | overrides


def _synthetic_disparity(n: int = N_DISPARITY_FRAMES) -> np.ndarray:
    """Depth-like uint8 disparity: a ramp that moves, plus an invalid block.

    Values stay in `1..max_disparity`; `0` is reserved for the §21.4 invalid
    region, which must never become a 0 m reading.
    """
    width, height = DISPARITY_SIZE
    ramp = (1 + (np.arange(width) * (MAX_DISPARITY[False] - 1)) // width).astype(
        np.uint8
    )
    frames = np.stack([
        np.roll(np.broadcast_to(ramp, (height, width)), k, 1) for k in range(n)
    ])
    frames[:, :8, :8] = 0

    return np.ascontiguousarray(frames)


def _encode(
    path: Path, frames: np.ndarray, *, pix_fmt: str, codec: str = "ffv1"
) -> Path:
    """Encode with imageio-ffmpeg's vendored binary (the system one is broken)."""
    n, height, width = frames.shape
    del n
    subprocess.run(  # noqa: S603
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-s",
            f"{width}x{height}",
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-c:v",
            codec,
            "-pix_fmt",
            pix_fmt,
            path.as_posix(),
        ],
        input=frames.tobytes(),
        check=True,
    )

    return path


# ------------------------------------------------------- synthetic MCAP episode

_FIELD = descriptor_pb2.FieldDescriptorProto  # ty: ignore[unresolved-attribute]
_SCHEMAS: dict[str, tuple[tuple[str, int], ...]] = {
    "Transform": tuple(
        (name, _FIELD.TYPE_DOUBLE) for name in ("x", "y", "z", "qx", "qy", "qz", "qw")
    ),
    "Orientation": tuple(
        (name, _FIELD.TYPE_DOUBLE) for name in ("qx", "qy", "qz", "qw")
    ),
    "ImageFrameIndex": (
        ("frame_index", _FIELD.TYPE_UINT32),
        ("sequence_number", _FIELD.TYPE_UINT64),
        ("device_timestamp_ns", _FIELD.TYPE_UINT64),
        ("host_arrival_time_ns", _FIELD.TYPE_UINT64),
    ),
    "SensorFrameTiming": (
        ("device_id", _FIELD.TYPE_UINT32),
        ("group_id", _FIELD.TYPE_UINT32),
        ("device_timestamp_us", _FIELD.TYPE_UINT64),
        ("host_arrival_time_ns", _FIELD.TYPE_UINT64),
    ),
}


def _proto_types() -> tuple[bytes, dict[str, type]]:
    """The §2 message shapes, built without protoc so the fixture is hermetic."""
    file = descriptor_pb2.FileDescriptorProto(  # ty: ignore[unresolved-attribute]
        name="nero_recording.proto", package="nero", syntax="proto3"
    )
    for name, fields in _SCHEMAS.items():
        message = file.message_type.add()
        message.name = name
        for number, (field_name, field_type) in enumerate(fields, start=1):
            field = message.field.add()
            field.name, field.number = field_name, number
            field.type, field.label = field_type, _FIELD.LABEL_OPTIONAL

    pool = descriptor_pool.DescriptorPool()  # ty: ignore[possibly-missing-implicit-call]
    pool.Add(file)

    return (
        descriptor_pb2.FileDescriptorSet(file=[file]).SerializeToString(),  # ty: ignore[unresolved-attribute]
        {
            name: message_factory.GetMessageClass(
                pool.FindMessageTypeByName(f"nero.{name}")
            )
            for name in _SCHEMAS
        },
    )


def _write_episode(
    path: Path,
    *,
    declaration: Mapping[str, str] | None,
    n_camera: int = N_CAMERA_FRAMES,
    n_disparity: int = N_DISPARITY_FRAMES,
    n_glove: int = 200,
) -> Path:
    """A minimal right-hand, single-camera episode with a disparity stream.

    Device clocks are exact and the host clock is device + a constant latency +
    non-negative jitter whose minimum is zero, which is what §3.2's
    minimum-latency estimator is defined against.
    """
    descriptor_set, types = _proto_types()
    rng = np.random.default_rng(0)

    def jitter(n: int) -> np.ndarray:
        out = rng.integers(0, 8_000_000, size=n)
        out[0] = 0

        return out

    with path.open("wb") as f:
        writer = Writer(f)
        writer.start()

        schema_ids = {
            name: writer.register_schema(
                name=f"nero.{name}", encoding="protobuf", data=descriptor_set
            )
            for name in _SCHEMAS
        }

        def channel(topic: str, schema: str) -> int:
            return writer.register_channel(
                topic=topic, message_encoding="protobuf", schema_id=schema_ids[schema]
            )

        # glove: `timing.rgmp` plus one sample per pose topic, ordinally joined
        glove_true = _T0_NS + np.arange(n_glove) * _GLOVE_PERIOD_NS
        glove_host = glove_true + _LATENCY_NS + jitter(n_glove)
        timing = channel("timing.rgmp", "SensorFrameTiming")
        for i in range(n_glove):
            message = types["SensorFrameTiming"](
                device_id=1,
                group_id=0,
                device_timestamp_us=(glove_true[i] + _GLOVE_EPOCH_NS) // 1_000,
                host_arrival_time_ns=glove_host[i],
            )
            writer.add_message(
                timing, glove_host[i], message.SerializeToString(), glove_host[i]
            )

        pose_topics = [
            "right.arm_sensor.coil_pro.transform",
            *(f"right.{finger}_finger_sensor.hub.transform" for finger in FINGERS),
        ]
        for t, topic in enumerate(pose_topics):
            channel_id = channel(topic, "Transform")
            for i in range(n_glove):
                message = types["Transform"](
                    x=0.1 * t + 1e-3 * i,
                    y=0.2 * t,
                    z=0.3 * t,
                    qx=0.0,
                    qy=0.0,
                    qz=0.0,
                    qw=1.0,
                )
                writer.add_message(
                    channel_id,
                    glove_host[i],
                    message.SerializeToString(),
                    glove_host[i],
                )

        channel_id = channel("right.hub.LTP_NED.orientation", "Orientation")
        for i in range(n_glove):
            message = types["Orientation"](qx=0.0, qy=0.0, qz=0.0, qw=1.0)
            writer.add_message(
                channel_id, glove_host[i], message.SerializeToString(), glove_host[i]
            )

        # `ImageFrameIndex` streams: the camera, and the disparity stream on its
        # own index, starting `DISPARITY_START_FRAME` camera frames later (§3).
        def frame_index_stream(topic: str, n: int, start_frame: int) -> None:
            channel_id = channel(topic, "ImageFrameIndex")
            true = (
                _T0_NS + 20_000_000 + (start_frame + np.arange(n)) * _CAMERA_PERIOD_NS
            )
            host = true + _LATENCY_NS + jitter(n)
            for k in range(n):
                message = types["ImageFrameIndex"](
                    frame_index=k,
                    sequence_number=k,
                    device_timestamp_ns=int(true[k] + _CAMERA_EPOCH_NS),
                    host_arrival_time_ns=int(host[k]),
                )
                writer.add_message(
                    channel_id, int(host[k]), message.SerializeToString(), int(host[k])
                )

        frame_index_stream("observation.images.base", n_camera, 0)
        frame_index_stream(
            f"{DISPARITY_TOPIC_PREFIX}base", n_disparity, DISPARITY_START_FRAME
        )

        writer.add_metadata(
            "episode",
            {"task": "synthetic", "active_hand": "right", "episode_id": "synthetic"},
        )
        if declaration is not None:
            writer.add_metadata(f"{DISPARITY_METADATA_PREFIX}base", dict(declaration))

        writer.finish()

    return path


@pytest.fixture(scope="session")
def disparity_frames() -> np.ndarray:
    return _synthetic_disparity()


@pytest.fixture(scope="session")
def disparity_mkv(
    tmp_path_factory: pytest.TempPathFactory, disparity_frames: np.ndarray
) -> Path:
    """§21.2: 8-bit `gray` FFV1 in MKV, exactly what the recorder will write."""
    return _encode(
        tmp_path_factory.mktemp("disparity") / "base_disparity.mkv",
        disparity_frames,
        pix_fmt="gray",
    )


@pytest.fixture(scope="session")
def disparity_mkv_measured(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Disparity at the MEASURED `stereo.image_size`, so the size guard's
    *passing* branch and the `calibration_path` resolution both get exercised at
    the resolution the real rig will use."""
    width, height = 1280, 800
    # 1 .. max_disparity inclusive, so both range endpoints are actually present
    ramp = (1 + (np.arange(width) * MAX_DISPARITY[False]) // width).astype(np.uint8)
    frames = np.ascontiguousarray(
        np.stack([
            np.roll(np.broadcast_to(ramp, (height, width)), k, 1) for k in range(4)
        ])
    )
    frames[:, :16, :16] = 0

    return _encode(
        tmp_path_factory.mktemp("measured") / "base_disparity.mkv",
        frames,
        pix_fmt="gray",
    )


def test_depth_through_the_shipped_calibration_path(
    disparity_mkv_measured: Path,
) -> None:
    """The path the config template takes: `camera` + `calibration_path`."""
    source = NeroArmsDisparityFrameSource(
        source=disparity_mkv_measured,
        output=DisparityOutput.depth,
        camera="base",
        calibration_path=CALIBRATION_PATH,
    )
    disparity = NeroArmsDisparityFrameSource(source=disparity_mkv_measured)[0]
    depth = source[0]
    valid = NeroArmsDisparityFrameSource(
        source=disparity_mkv_measured, output=DisparityOutput.valid
    )[0]

    assert depth.shape == (1, 800, 1280)
    assert torch.isfinite(depth).all()
    assert (valid.numpy() == (disparity.numpy() > 0)).all()

    levels = disparity.numpy()[0]

    assert round(float(depth.numpy()[0][levels == MAX_DISPARITY[False]].max()), 3) == (
        MIN_DEPTH_M
    ), "disparity 95 must land on the documented MinZ"
    assert round(float(depth.numpy()[0][levels == 1].max()), 1) == MAX_DEPTH_M
    assert (depth.numpy()[0][levels == 0] == 0.0).all()  # noqa: RUF069


def _builder(disparity_cameras: Sequence[str] = ()) -> NeroArmsDataFrameBuilder:
    return NeroArmsDataFrameBuilder(
        cameras=["base"],
        reference_camera="base",
        sides=["right"],
        disparity_cameras=disparity_cameras,
    )


# ------------------------------------------------------------ §21.1 declaration


def test_stereo_calibration_rejects_subpixel() -> None:
    """§21.4: an 8-bit read of subpixel disparity is wrong by a factor of 8."""
    with pytest.raises(ValidationError, match="subpixel"):
        _stereo(subpixel=True, max_disparity=760)


def test_declaration_rejects_subpixel() -> None:
    with pytest.raises(ValidationError, match="subpixel"):
        DisparityDeclaration.from_mcap_metadata(
            _declaration(subpixel="true", max_disparity="760")
        )


def test_declaration_rejects_max_disparity_contradiction() -> None:
    """§21.3: assuming 95 when extended disparity was on halves every depth."""
    with pytest.raises(ValidationError, match="max_disparity"):
        DisparityDeclaration.from_mcap_metadata(_declaration(extended_disparity="true"))

    with pytest.raises(ValidationError, match="max_disparity"):
        _stereo(extended_disparity=True)


def test_declaration_accepts_extended_disparity() -> None:
    declaration = DisparityDeclaration.from_mcap_metadata(
        _declaration(extended_disparity="true", max_disparity=str(MAX_DISPARITY[True]))
    )

    assert declaration.max_disparity == MAX_DISPARITY[True] == EXTENDED_MAX_DISPARITY
    assert not declaration.subpixel
    # §21.3: declared, never inferred -- including the mono pair used
    assert declaration.mono_sockets == "CAM_B,CAM_C"


def test_declaration_must_disagree_loudly_with_calibration() -> None:
    declaration = DisparityDeclaration.from_mcap_metadata(_declaration())
    declaration.check_against(_stereo(), camera="base")

    with pytest.raises(ValueError, match="disagrees"):
        declaration.check_against(_stereo(image_size=(1280, 800)), camera="base")


def test_shipped_calibration_carries_the_measured_mono_pair() -> None:
    """§21.5, now measured: the MONO pair, not the RGB `CAM_A` above it."""
    stereo = NeroArmsCalibration.from_path(CALIBRATION_PATH).stereo_for("base")

    assert (stereo.fx_mono, stereo.baseline_m) == (FX_MONO_BASE, BASELINE_M)
    assert stereo.image_size == (1280, 800), "`fx_mono` is expressed at this size"
    assert stereo.mono_sockets == ("CAM_B", "CAM_C")
    assert not stereo.placeholder
    # the RGB CAM_A fx is 1152.2 -- using it would scale every depth by 2x
    assert (
        stereo.fx_mono
        != NeroArmsCalibration.from_path(CALIBRATION_PATH).cameras["base"].intrinsics.fx
    )


def test_measured_stereo_reproduces_the_documented_range() -> None:
    """`CAMERA_RIG.md` §4.2: MinZ must be checked against the mount height."""
    stereo = NeroArmsCalibration.from_path(CALIBRATION_PATH).stereo_for("base")

    assert round(stereo.min_depth_m, 3) == MIN_DEPTH_M, "MinZ, default mode"
    assert round(stereo.max_depth_m, 1) == MAX_DEPTH_M, "far limit at disparity 1"

    extended = stereo.model_copy(
        update={"max_disparity": MAX_DISPARITY[True], "extended_disparity": True}
    )

    assert round(extended.min_depth_m, 3) == MIN_DEPTH_EXTENDED_M, (
        "MinZ, extended disparity"
    )

    # the same numbers through the conversion the ingestion actually uses
    depth, valid = disparity_to_depth(
        torch.tensor([0, 1, stereo.max_disparity], dtype=torch.uint8), stereo
    )

    assert np.allclose(depth.numpy(), [0.0, stereo.max_depth_m, stereo.min_depth_m])
    assert (valid.numpy() == [False, True, True]).all()


def _without_stereo(tmp_path: Path) -> Path:
    """The pre-measurement shape of the sidecar: no `stereo:` block at all."""
    calibration = yaml.safe_load(CALIBRATION_PATH.read_text(encoding="utf-8"))
    del calibration["stereo"]
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(calibration), encoding="utf-8")

    return path


def test_absent_stereo_block_is_refused(tmp_path: Path) -> None:
    """§21.4: no safe default -- a guessed `fx_mono` scales every depth."""
    calibration = NeroArmsCalibration.from_path(_without_stereo(tmp_path))

    assert calibration.stereo == {}

    with pytest.raises(ValueError, match="no `stereo` calibration"):
        calibration.stereo_for("base")


def test_placeholder_stereo_is_refused() -> None:
    calibration = NeroArmsCalibration.from_path(CALIBRATION_PATH).model_copy(deep=True)
    calibration.stereo = {"base": _stereo(placeholder=True)}

    with pytest.raises(ValueError, match="PLACEHOLDER"):
        calibration.stereo_for("base")


# -------------------------------------------------------------- §21.4 conversion


def test_disparity_to_depth_matches_hand_computed_values() -> None:
    disparity = torch.tensor([[0, 1, 2, 30, 95]], dtype=torch.uint8)
    depth, valid = disparity_to_depth(disparity, _stereo())

    # fx_mono * baseline_m == 400 * 0.075 == 30.0 m*px
    assert np.allclose(depth.numpy(), [[0.0, 30.0, 15.0, 1.0, 30.0 / 95.0]], atol=1e-6)
    assert (valid.numpy() == [[False, True, True, True, True]]).all()


def test_invalid_disparity_is_masked_never_zero_metres() -> None:
    """§21.4: `disparity == 0` is INVALID, not zero distance."""
    disparity = torch.zeros((4, 4), dtype=torch.uint8)
    depth, valid = disparity_to_depth(disparity, _stereo())

    assert not valid.any(), "every pixel is invalid"
    assert torch.isfinite(depth).all(), "no inf: the denominator is clamped first"
    assert not torch.isnan(depth * valid).any(), "inf * 0 would be nan"
    # the value is 0.0 *and* masked -- a consumer that ignores the mask must not
    # be able to mistake it for a valid reading, which is what the mask is for
    assert (depth == 0.0).all()  # noqa: RUF069  -- exactly the constant written above


# ------------------------------------------------------------------ §21.2 decode


def test_ffv1_gray8_decodes_bit_exact(
    disparity_mkv: Path, disparity_frames: np.ndarray
) -> None:
    source = NeroArmsDisparityFrameSource(source=disparity_mkv)

    assert len(source) == N_DISPARITY_FRAMES

    single = source[3]

    assert single.shape == (1, DISPARITY_SIZE[1], DISPARITY_SIZE[0])
    assert single.dtype == torch.uint8
    assert np.array_equal(single.numpy()[0], disparity_frames[3])

    batch = source[[0, 7, 11]]

    assert batch.shape == (3, 1, DISPARITY_SIZE[1], DISPARITY_SIZE[0])
    assert np.array_equal(batch.numpy()[:, 0], disparity_frames[[0, 7, 11]])


def test_sixteen_bit_is_refused_not_silently_downconverted(tmp_path: Path) -> None:
    """§21.2: torchcodec does not raise on 16-bit -- it returns wrong data."""
    frames = _synthetic_disparity(4).astype("<u2") * 8  # as subpixel would store it
    path = _encode(tmp_path / "d16.mkv", frames, pix_fmt="gray16le")

    with pytest.raises(ValueError, match="PyAV"):
        NeroArmsDisparityFrameSource(source=path)

    # the failure mode being guarded against, demonstrated
    decoded = VideoDecoder(path.as_posix()).get_frame_at(index=0).data

    assert decoded.dtype == torch.uint8, "torchcodec silently down-converts"
    assert not np.array_equal(decoded.numpy()[0], frames[0])


def test_lossy_codec_is_refused(tmp_path: Path) -> None:
    """§21.2: libx264 measured a max error of 81 levels at 95 full scale."""
    path = _encode(
        tmp_path / "lossy.mkv", _synthetic_disparity(4), pix_fmt="gray", codec="libx264"
    )

    with pytest.raises(ValueError, match="lossless"):
        NeroArmsDisparityFrameSource(source=path)


def test_depth_source_emits_metres_and_mask(
    disparity_mkv: Path, disparity_frames: np.ndarray
) -> None:
    stereo = _stereo()
    depth = NeroArmsDisparityFrameSource(
        source=disparity_mkv, output=DisparityOutput.depth, stereo=stereo
    )[[0, 5]]
    valid = NeroArmsDisparityFrameSource(
        source=disparity_mkv, output=DisparityOutput.valid
    )[[0, 5]]

    assert depth.dtype == torch.float32
    assert valid.dtype == torch.bool
    assert torch.isfinite(depth).all()

    reference = disparity_frames[[0, 5]].astype(np.float64)
    invalid = reference == 0

    assert (valid.numpy()[:, 0] == ~invalid).all()
    assert (depth.numpy()[:, 0][invalid] == 0.0).all()  # noqa: RUF069
    assert np.allclose(
        depth.numpy()[:, 0][~invalid],
        (FX_MONO * BASELINE_M / reference[~invalid]),
        atol=1e-5,
    )


def test_depth_without_stereo_calibration_fails_loudly(
    disparity_mkv: Path, tmp_path: Path
) -> None:
    """§21.4: without the mono `fx`/baseline, refuse -- never guess."""
    with pytest.raises(ValueError, match="no `stereo` calibration"):
        NeroArmsDisparityFrameSource(
            source=disparity_mkv,
            output=DisparityOutput.depth,
            camera="base",
            calibration_path=_without_stereo(tmp_path),
        )

    with pytest.raises(ValueError, match="calibration_path"):
        NeroArmsDisparityFrameSource(source=disparity_mkv, output=DisparityOutput.depth)

    # raw disparity must still work with no calibration at all
    assert NeroArmsDisparityFrameSource(source=disparity_mkv)[0].dtype == torch.uint8


def test_depth_rejects_a_resolution_mismatch(disparity_mkv: Path) -> None:
    """`fx_mono` describes one resolution; anything else scales every depth."""
    with pytest.raises(ValueError, match="image_size"):
        NeroArmsDisparityFrameSource(
            source=disparity_mkv,
            output=DisparityOutput.depth,
            stereo=_stereo(image_size=(1280, 800)),
        )


# ------------------------------------------------------------------ §21.4 join


@pytest.fixture(scope="session")
def calibration_no_stereo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The join tests care about indexes, not about metric depth."""
    return _without_stereo(tmp_path_factory.mktemp("calibration"))


@pytest.fixture(scope="session")
def disparity_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_episode(
        tmp_path_factory.mktemp("episode") / "data.mcap", declaration=_declaration()
    )


def test_builder_joins_disparity_on_its_own_frame_index(
    disparity_episode: Path, calibration_no_stereo: Path
) -> None:
    """§3/§21.4: the disparity stream has its own, shorter, index."""
    df = _builder(["base"])(disparity_episode, calibration_no_stereo)

    assert "frame_index.disparity.base" in df.columns
    assert "goal.frame_index.disparity.base" in df.columns

    base = df["frame_index.base"].to_numpy()
    disparity = df["frame_index.disparity.base"].to_numpy()

    assert disparity.max() < N_DISPARITY_FRAMES < N_CAMERA_FRAMES
    # the stream starts two camera frames late, so its index trails by two --
    # anything that assumed a shared index would fail here
    late = base >= DISPARITY_START_FRAME

    assert (disparity[late] == base[late] - DISPARITY_START_FRAME).all()
    assert (disparity[~late] == 0).all(), "clipped, never negative or extrapolated"
    assert int(df["goal.frame_index.disparity.base"][0]) == N_DISPARITY_FRAMES - 1


def test_builder_without_disparity_is_unchanged(
    disparity_episode: Path, calibration_no_stereo: Path
) -> None:
    """The depth stream is optional and off by default (§8)."""
    df = _builder()(disparity_episode, calibration_no_stereo)

    assert not [c for c in df.columns if "disparity" in c]
    assert df["frame_index.base"].is_sorted(descending=False)


def test_builder_requires_the_declaration(
    tmp_path: Path, calibration_no_stereo: Path
) -> None:
    """§17.1: presence is declared, never inferred from absence."""
    path = _write_episode(tmp_path / "data.mcap", declaration=None)

    with pytest.raises(ValueError, match="does not declare"):
        _builder(["base"])(path, calibration_no_stereo)

    # ... and without disparity ingestion the same episode is fine
    assert len(_builder()(path, calibration_no_stereo)) > 0


def test_builder_rejects_declared_subpixel(
    tmp_path: Path, calibration_no_stereo: Path
) -> None:
    path = _write_episode(
        tmp_path / "data.mcap",
        declaration=_declaration(subpixel="true", max_disparity="760"),
    )

    with pytest.raises(ValidationError, match="subpixel"):
        _builder(["base"])(path, calibration_no_stereo)


def test_builder_cross_checks_the_declaration_against_calibration(
    disparity_episode: Path,
) -> None:
    """§21.3: the episode declares 64x48; the measured rig says 1280x800."""
    with pytest.raises(ValueError, match="disagrees"):
        _builder(["base"])(disparity_episode, CALIBRATION_PATH)
