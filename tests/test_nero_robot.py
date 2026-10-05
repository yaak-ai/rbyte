"""Robot-native nero ingestion for the patch family (P4/P6/P10, contract v3).

Self-contained on a synthetic MCAP; the convert.py cross-check runs on a pulled
robot episode and against nutron-cli's own `convert.align` when both are present
(`NUTRON_PULLED_EPISODES`, `NUTRON_CLI_ROOT`), and skips otherwise.
"""

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.writer import Writer

from rbyte.samples.nero import _vendor  # ruff:ignore[import-private-name]
from rbyte.samples.nero._vendor import hand_features as hf  # ruff:ignore[import-private-name]
from rbyte.samples.nero.robot import (
    FPS,
    MAX_STALE_NS,
    PREV_NS,
    NeroRobotReader,
    NeroRobotWindowGrouper,
    build_rows,
    chunk_offsets_ns,
    grid_index,
    read_episode,
    zoh,
)

VENDOR_FILE = Path(_vendor.__file__).parent / "hand_features.py"
NUTRON_CLI = Path(
    os.environ.get("NUTRON_CLI_ROOT", "/home/max/Code/nutron-cli-patch-policy")
)
PULLED = Path(
    os.environ.get(
        "NUTRON_PULLED_EPISODES",
        str(Path.home() / ".config/nutron/episodes-pulled/accepted"),
    )
)

PERIOD_NS = 1_000_000_000 // FPS
T0 = 1_790_000_000_000_000_000

# ------------------------------------------------------------- vendored copy


def test_vendored_hand_features_matches_its_pins() -> None:
    digest = hashlib.sha256(VENDOR_FILE.read_bytes()).hexdigest()
    assert digest == _vendor.HAND_FEATURES_SHA256
    assert hf.HAND_FEATURES_VERSION == _vendor.HAND_FEATURES_VERSION
    assert hf.HAND_TOKEN_SPEC_VERSION == _vendor.HAND_TOKEN_SPEC_VERSION


def test_vendored_hand_features_is_byte_identical_to_nutron_cli() -> None:
    source = NUTRON_CLI / "runtime" / "jetson" / "hand_features.py"
    if not source.exists():
        pytest.skip(f"no nutron-cli checkout at {source}")
    assert VENDOR_FILE.read_bytes() == source.read_bytes(), (
        "vendored hand_features.py drifted: re-run hand_features_sync.py --vendor "
        "and update rbyte/samples/nero/_vendor/__init__.py"
    )


# ----------------------------------------------------------- synthetic MCAP

_F = descriptor_pb2.FieldDescriptorProto  # ty: ignore[unresolved-attribute]
_REP, _OPT = _F.LABEL_REPEATED, _F.LABEL_OPTIONAL
_SCHEMAS = {
    "JointState": (("positions", _F.TYPE_DOUBLE, _REP),),
    "FingerCommand": (("counts", _F.TYPE_INT32, _REP),),
    "ImageFrameIndex": (
        ("frame_index", _F.TYPE_UINT32, _OPT),
        ("device_timestamp_ns", _F.TYPE_UINT64, _OPT),
    ),
    "HandTactile": (
        ("motor_valid", _F.TYPE_BOOL, _OPT),
        ("sample_time_ns", _F.TYPE_UINT64, _OPT),
        ("positions", _F.TYPE_INT32, _REP),
        ("currents", _F.TYPE_INT32, _REP),
        ("touch_sample_time_ns", _F.TYPE_UINT64, _OPT),
        ("touch_finger_present", _F.TYPE_BOOL, _REP),
        ("normal_force", _F.TYPE_INT32, _REP),
        ("tangential_force", _F.TYPE_INT32, _REP),
    ),
}


def _types() -> tuple[bytes, dict[str, type]]:
    file = descriptor_pb2.FileDescriptorProto(  # ty: ignore[unresolved-attribute]
        name="nutron_recording_test.proto", package="nutron", syntax="proto3"
    )
    for name, fields in _SCHEMAS.items():
        message = file.message_type.add()
        message.name = name
        for number, (field_name, field_type, label) in enumerate(fields, start=1):
            field = message.field.add()
            field.name, field.number, field.type, field.label = (
                field_name,
                number,
                field_type,
                label,
            )
    pool = descriptor_pool.DescriptorPool()  # ty: ignore[possibly-missing-implicit-call]
    pool.Add(file)
    return (
        descriptor_pb2.FileDescriptorSet(file=[file]).SerializeToString(),  # ty: ignore[unresolved-attribute]
        {
            name: message_factory.GetMessageClass(
                pool.FindMessageTypeByName(f"nutron.{name}")
            )
            for name in _SCHEMAS
        },
    )


def _write(  # ruff:ignore[complex-structure, too-many-arguments, too-many-locals]
    path: Path,
    *,
    n_frames: int = 400,
    drop_frame: int | None = None,
    tactile: bool = True,
    tactile_gap: tuple[int, int] | None = None,
    reset_offset_ns: tuple[int, int] | None = None,
    outcome: str = "success",
) -> Path:
    """A left-hand robot take: 50 Hz arm, 45 Hz hand command, 30 fps cameras.

    Signals are deterministic functions of time, so chunk values can be checked
    against a direct ZOH. The base camera optionally drops one frame
    (`drop_frame`) and the tactile stream optionally goes silent over
    `tactile_gap` (frame indices).
    """
    descriptor, kinds = _types()
    path.mkdir(parents=True, exist_ok=True)
    mcap = path / "data.mcap"
    started = T0 - 500_000_000
    end = T0 + n_frames * PERIOD_NS
    with mcap.open("wb") as f:
        writer = Writer(f)
        writer.start()
        sid = {
            name: writer.register_schema(
                name=f"nutron.{name}", encoding="protobuf", data=descriptor
            )
            for name in _SCHEMAS
        }

        def channel(topic: str, schema: str) -> int:
            return writer.register_channel(
                topic=topic, message_encoding="protobuf", schema_id=sid[schema]
            )

        def put(ch: int, t: int, msg: object) -> None:
            writer.add_message(ch, int(t), msg.SerializeToString(), int(t))  # ty: ignore[unresolved-attribute]

        meas, cmd = (
            channel("robot.measured.q", "JointState"),
            channel("robot.command.q", "JointState"),
        )
        for i, t in enumerate(range(T0 - 400_000_000, end + 200_000_000, 20_000_000)):
            q = [np.sin(0.01 * i + j) for j in range(7)]
            put(meas, t + 3_000_000, kinds["JointState"](positions=q))
            put(cmd, t, kinds["JointState"](positions=[v + 0.01 for v in q]))
        hand = channel("robot.hand.command", "FingerCommand")
        for i, t in enumerate(range(T0 - 400_000_000, end + 200_000_000, 22_222_222)):
            counts = [(i * (j + 1)) % 1001 for j in range(6)]
            put(hand, t, kinds["FingerCommand"](counts=counts))
        if tactile:
            tac = channel("robot.hand.tactile", "HandTactile")
            for i, t in enumerate(range(T0 - 400_000_000, end, 95_000_000)):
                frame = (t - T0) // PERIOD_NS
                if tactile_gap and tactile_gap[0] <= frame < tactile_gap[1]:
                    continue
                put(
                    tac,
                    t + 5_000_000,
                    kinds["HandTactile"](
                        motor_valid=True,
                        sample_time_ns=t,
                        positions=[(i * 7 + j) % 1000 for j in range(6)],
                        currents=[(i * 3 + j) % 900 - 400 for j in range(6)],
                    ),
                )
        frame_idx = 0
        for camera in ("base", "side_left", "side_right"):
            ch = channel(f"observation.images.{camera}", "ImageFrameIndex")
            frame_idx = 0
            skew = {"base": 0, "side_left": 4_000_000, "side_right": -6_000_000}[camera]
            for k in range(n_frames):
                if camera == "base" and k == drop_frame:
                    continue
                put(
                    ch,
                    T0 + k * PERIOD_NS + skew,
                    kinds["ImageFrameIndex"](frame_index=frame_idx),
                )
                frame_idx += 1
        writer.add_metadata(
            "episode",
            {
                "task": "synthetic",
                "active_hand": "left",
                "recording_started_at_unix_ns": str(started),
            },
        )
        writer.add_metadata("episode_outcome", {"outcome": outcome})
        spans = [] if reset_offset_ns is None else [list(reset_offset_ns)]
        writer.add_metadata(
            "hand_reset", {"events": str(len(spans)), "spans_ns": json.dumps(spans)}
        )
        writer.finish()
    (path / "outcome.json").write_text(json.dumps({"outcome": outcome}))
    return mcap


@pytest.fixture(scope="module")
def episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write(tmp_path_factory.mktemp("robot") / "ep0")


def test_state_and_chunk_follow_the_convert_rules(episode: Path) -> None:
    ep = read_episode(episode)
    rows = build_rows(ep)
    c = rows.columns
    t = c["t_ns"]
    assert len(t) > 0
    side = 0  # left
    # state = measured ZOH at t + hand command ZOH at t - 1/30 s
    jm, _ = zoh(ep.measured_ts, t)
    jp, age = zoh(ep.hand_ts, t - PREV_NS)
    assert (age <= MAX_STALE_NS).all()
    np.testing.assert_array_equal(
        c["state"][:, side],
        np.concatenate([ep.measured[jm], ep.hand[jp]], axis=-1).astype(np.float32),
    )
    # chunk[k] = command ZOH at t + k/30 s (non-padded steps)
    tk = t[:, None] + chunk_offsets_ns(100)[None]
    jc, _ = zoh(ep.command_ts, tk.reshape(-1))
    jh, _ = zoh(ep.hand_ts, tk.reshape(-1))
    want = np.concatenate(
        [ep.command[jc].reshape(*tk.shape, 7), ep.hand[jh].reshape(*tk.shape, 6)],
        axis=-1,
    ).astype(np.float32)
    real = ~c["action.is_pad"]
    np.testing.assert_array_equal(c["action.chunk"][:, :, side][real], want[real])
    # k = 0 is the command at t itself, and hand_prev is NOT a copy of it
    j0, _ = zoh(ep.hand_ts, t)
    np.testing.assert_array_equal(
        c["action.chunk"][:, 0, side, 7:], ep.hand[j0].astype(np.float32)
    )
    assert not np.array_equal(
        c["action.chunk"][:, 0, side, 7:], c["state"][:, side, 7:]
    )
    # the invalid side is exactly zero and masked
    assert (c["side_valid"] == [True, False]).all()
    assert not c["state"][:, 1].any()
    assert not c["action.chunk"][:, :, 1].any()


def test_padding_holds_the_last_command_at_the_run_end(episode: Path) -> None:
    ep = read_episode(episode)
    rows = build_rows(ep)
    c = rows.columns
    pad = c["action.is_pad"]
    assert not pad[:, 0].any()
    # suffix-shaped, capped at chunk/2, and present at the end of the run
    first = np.where(pad.any(1), pad.argmax(1), 100)
    for i in range(len(pad)):
        assert pad[i, first[i] :].all()
    assert pad.sum(1).max() <= 50  # ruff:ignore[magic-value-comparison]
    assert pad[-1].sum() == 50  # ruff:ignore[magic-value-comparison]
    # held value = the last real step
    i = len(pad) - 1
    k = first[i]
    np.testing.assert_array_equal(
        c["action.chunk"][i, k:, 0],
        np.broadcast_to(c["action.chunk"][i, k - 1, 0], (100 - k, 13)),
    )
    # t + k/30 for the last real step lies within the run; the first pad beyond it
    ((_, stop),) = rows.stats["runs"]
    t_end = ep.cam_ts["base"][stop - 1]
    assert c["t_ns"][i] + chunk_offsets_ns(100)[k - 1] <= t_end
    assert c["t_ns"][i] + chunk_offsets_ns(100)[k] > t_end


def test_hand_token_columns_equal_the_shared_builder(episode: Path) -> None:
    ep = read_episode(episode)
    rows = build_rows(ep)
    c = rows.columns
    motor = hf.motor_timeline(ep.tactile)
    commands = hf.make_command_timeline(ep.hand_ts, ep.hand_counts)
    groups = ("current", "pos_err", "pos")
    for i in range(0, len(c["t_ns"]), 37):
        vec, valid, _ = hf.build_token(
            int(c["t_ns"][i]),
            motor,
            hf.trim_commands_token(commands, int(c["t_ns"][i])),
            groups,
        )
        composed = np.concatenate([
            c["hand.current"][i],
            c["hand.pos_err"][i],
            c["hand.pos"][i],
            [c["hand.age"][i], float(c["hand.motor_ok"][i])],
        ]).astype(np.float32)
        if not c["hand.motor_ok"][i]:
            composed[:] = 0
        assert valid == bool(c["hand.motor_ok"][i])
        np.testing.assert_array_equal(vec, composed)


def test_a_hand_refusal_keeps_the_frame(tmp_path: Path) -> None:
    """No NULL-row deletion: a stale hand is a `no_hand` frame, not a dropped one."""
    mcap = _write(tmp_path / "gap", tactile_gap=(100, 160))
    with_gap = build_rows(read_episode(mcap))
    without = build_rows(read_episode(_write(tmp_path / "nogap")))
    np.testing.assert_array_equal(with_gap.columns["t_ns"], without.columns["t_ns"])
    refused = ~with_gap.columns["hand.motor_ok"]
    assert refused.sum() > 40  # ruff:ignore[magic-value-comparison]
    for name in ("hand.current", "hand.pos_err", "hand.pos"):
        assert not with_gap.columns[name][refused].any()


def test_a_failed_take_is_refused(tmp_path: Path) -> None:
    # convert.py's rule: only takes whose MCAP outcome AND outcome.json say success
    with pytest.raises(ValueError, match="not success"):
        read_episode(_write(tmp_path / "failed", n_frames=60, outcome="failure"))
    mcap = _write(tmp_path / "relabeled", n_frames=60)
    (mcap.parent / "outcome.json").write_text(json.dumps({"outcome": "failure"}))
    with pytest.raises(ValueError, match=r"outcome\.json"):
        read_episode(mcap)


def test_a_reset_span_excludes_rows(tmp_path: Path) -> None:
    # offsets from recording start (= T0 - 0.5 s): a span at frames ~ 150..151
    start = 500_000_000 + 150 * PERIOD_NS
    rows = build_rows(
        read_episode(
            _write(
                tmp_path / "reset",
                n_frames=700,
                reset_offset_ns=(start, start + PERIOD_NS),
            )
        ),
        min_run=30,
    )
    t = rows.columns["t_ns"]
    span_lo = T0 + 150 * PERIOD_NS - hf.RESET_PAD_BEFORE_NS
    span_hi = T0 + 151 * PERIOD_NS + hf.RESET_PAD_AFTER_NS
    assert not ((t >= span_lo) & (t <= span_hi)).any()
    # chunks of rows before the span are padded before reaching into it
    before = t < span_lo
    tk = t[before, None] + chunk_offsets_ns(100)[None]
    real = ~rows.columns["action.is_pad"][before]
    assert (tk[real] < span_lo).all()


def test_grid_index_marks_a_dropped_frame() -> None:
    t = (
        T0
        + np.array([0, 1, 2, 4, 5]) * PERIOD_NS
        + np.array([0, 2, -3, 1, 0]) * 1_000_000
    )
    np.testing.assert_array_equal(grid_index(t), [0, 1, 2, 4, 5])


# ------------------------------------------------------------------ windows


def test_windows_are_uniform_and_skip_gaps(tmp_path: Path) -> None:
    frame = NeroRobotReader()(_write(tmp_path / "drop", drop_frame=200))
    grid = frame["grid_index"].to_numpy()
    assert 200 not in grid  # ruff:ignore[magic-value-comparison]  (the dropped base frame never became a row)
    windows = NeroRobotWindowGrouper(clip_frames=8, episode_stride=7)(frame)
    g = np.stack(windows["grid_index"].to_numpy())  # ty:ignore[no-matching-overload]
    assert (np.diff(g, axis=1) == 3).all()  # ruff:ignore[magic-value-comparison]
    # no window spans the dropped frame, even between two gathered frames
    assert not ((g[:, :1] <= 200) & (g[:, -1:] >= 200)).any()  # ruff:ignore[magic-value-comparison]
    # all three 30 Hz phases are used
    assert set((g[:, 0] % 3).tolist()) == {0, 1, 2}
    # per-frame columns gained the T axis, constants did not
    assert windows["state"].dtype.shape == (8, 2, 13)  # ty:ignore[unresolved-attribute]
    assert windows["action.chunk"].dtype.shape == (8, 100, 2, 13)  # ty:ignore[unresolved-attribute]
    assert windows["side_valid"].dtype.shape == (2,)  # ty:ignore[unresolved-attribute]


def test_window_starts_require_every_intermediate_row() -> None:
    builder = NeroRobotWindowGrouper(clip_frames=3, frame_stride=3)
    grid = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15])
    starts = builder.starts(grid)
    first = grid[starts]
    # a window [g, g + 6] must not contain 7 (missing)
    assert not ((first <= 7) & (first + 6 >= 7)).any()  # ruff:ignore[magic-value-comparison]
    assert first.tolist() == [0, 8, 9]


# ---------------------------------------------- convert.py cross-check (real)


def _convert():  # ruff:ignore[missing-return-type-private-function]
    training = NUTRON_CLI / "runtime" / "training"
    if not (training / "convert.py").exists():
        pytest.skip("no nutron-cli checkout")
    try:
        import recording_pb2  # ruff:ignore[unused-import, import-outside-top-level]  # ty:ignore[unresolved-import]
    except ImportError:
        sys.path.insert(0, str(NUTRON_CLI / "runtime"))
    sys.path.insert(0, str(training))
    if "cv2" not in sys.modules:  # convert only needs cv2 for mp4 I/O
        stub = types.ModuleType("cv2")

        class _Cap:
            def __init__(self, *_: object) -> None: ...
            def get(self, *_: object) -> int:  # ruff:ignore[no-self-use]
                return 0

            def release(self) -> None: ...

        stub.VideoCapture = _Cap  # ty: ignore[unresolved-attribute]
        stub.CAP_PROP_FRAME_COUNT = 7  # ty: ignore[unresolved-attribute]
        sys.modules["cv2"] = stub
    try:
        import convert  # ruff:ignore[import-outside-top-level]  # ty:ignore[unresolved-import]
    except ImportError as exc:
        pytest.skip(f"convert.py not importable here: {exc}")
    return convert


@pytest.mark.parametrize(
    "name",
    sorted(p.name for p in PULLED.glob("*") if (p / "data.mcap").exists())
    or ["<none>"],
)
def test_bit_identical_to_convert_on_a_pulled_episode(name: str) -> None:
    if name == "<none>":
        pytest.skip(f"no pulled episodes under {PULLED}")
    convert = _convert()
    ref = convert.read_mcap(PULLED / name)
    a = convert.align(ref)
    ep = read_episode(PULLED / name / "data.mcap")
    rows = build_rows(ep)
    c = rows.columns
    i = rows.keep
    assert a.valid[i].all()
    want_state = np.concatenate(
        (ref.measured.val[a.j_measured[i]], ref.hand.val[a.j_hand_prev[i]]), axis=-1
    ).astype(np.float32)
    want_action = np.concatenate(
        (ref.command.val[a.j_command[i]], ref.hand.val[a.j_hand[i]]), axis=-1
    ).astype(np.float32)
    np.testing.assert_array_equal(c["state"][:, 0], want_state)
    np.testing.assert_array_equal(c["action.chunk"][:, 0, 0], want_action)
    for camera in ("side_left", "side_right"):
        np.testing.assert_array_equal(
            c[f"frame_index.{camera}"], ref.cams[camera].idx[a.side_pick[camera][i]]
        )
    np.testing.assert_array_equal(c["frame_index.base"], ref.cams["base"].idx[i])
    # and the hand token blocks equal convert's own builder inputs
    h = convert.hand_env(ref)
    np.testing.assert_array_equal(
        c["hand.motor_ok"],
        hf.build_tokens(
            ref.cams["base"].ts[i],
            hf.motor_timeline(ref.tactile or []),
            hf.make_command_timeline(ref.hand.ts, ref.hand_counts),
        ).motor_ok,
    )
    assert (~h.reset_poison[i]).all()
