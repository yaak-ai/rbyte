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
    ROBOT_SIDES,
    SIDE_TOL_NS,
    NeroRobotReader,
    NeroRobotWindowGrouper,
    align,
    build_rows,
    chunk_offsets_ns,
    grid_index,
    layout_names,
    read_episode,
    rows_to_frame,
    runs_of,
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

#: the 85-take bimanual corpus (read-only); parity/probe tests skip without it
BIMANUAL = Path(
    os.environ.get(
        "NUTRON_BIMANUAL_EPISODES", "/nasa/drives/nero-arms/cube-bimanual/2026-10-07"
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
        ("side", _F.TYPE_STRING, _OPT),
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


#: a bimanual take's recorder metadata (signal_groups.bimanual_contract_metadata)
BIMANUAL_META = {
    "capture_mode": "bus_bimanual",
    "active_hand": "both",
    "action_dimension": "26",
    "state_dimension": "26",
    "action_order": ",".join(
        f"{s}.{part}.{i}"
        for s in ("left", "right")
        for part, n in (("arm", 7), ("hand", 6))
        for i in range(n)
    ),
    "state_order": ",".join(
        f"{s}.{part}.{i}"
        for s in ("left", "right")
        for part, n in (("arm", 7), ("hand", 6))
        for i in range(n)
    ),
    "arm_units": "radians",
    "hand_units": "revo2_counts_0_to_1000",
    "hand_channel_order": "thumb_flex,thumb_aux,index,middle,ring,pinky",
    "state_hand_source": "tactile positions",  # ignored: hand_prev everywhere
}


def _write(  # ruff:ignore[complex-structure, too-many-arguments, too-many-locals, too-many-statements]
    path: Path,
    *,
    n_frames: int = 400,
    drop_frame: int | None = None,
    tactile: bool = True,
    tactile_gap: tuple[int, int] | None = None,
    reset_offset_ns: tuple[int, int] | None = None,
    outcome: str = "success",
    bimanual: bool = False,
    episode_meta: dict[str, str] | None = None,
    right_end_frames_early: int = 0,
    tactile_gap_side: str = "left",
    reset_side: str = "left",
    outcome_reset: dict | None = None,
    hand_count_offset: int = 0,
    tactile_side_tag: dict[str, str] | None = None,
) -> Path:
    """A robot take: 50 Hz arm, 45 Hz hand command, 30 fps cameras.

    Single-arm (default): a left-hand take on the `robot.*` topics. Signals are
    deterministic functions of time, so chunk values can be checked against a
    direct ZOH. The base camera optionally drops one frame (`drop_frame`) and the
    tactile stream optionally goes silent over `tactile_gap` (frame indices).

    `bimanual=True`: a `bus_bimanual` take on `robot.{left,right}.*` with
    DIFFERENT signals per side (a left/right swap cannot pass). The right arm's
    streams can stop `right_end_frames_early` frames before the left's; the
    tactile gap and the recorder reset span apply to one side
    (`tactile_gap_side`, `reset_side`, as MCAP `hand_reset.{side}`).
    """
    descriptor, kinds = _types()
    path.mkdir(parents=True, exist_ok=True)
    mcap = path / "data.mcap"
    started = T0 - 500_000_000
    end = T0 + n_frames * PERIOD_NS
    sides = ("left", "right") if bimanual else (None,)
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

        for side in sides:
            prefix = "robot." if side is None else f"robot.{side}."
            r = side == "right"  # the right arm gets its own signals
            side_end = end - (right_end_frames_early * PERIOD_NS if r else 0)
            meas, cmd = (
                channel(f"{prefix}measured.q", "JointState"),
                channel(f"{prefix}command.q", "JointState"),
            )
            for i, t in enumerate(
                range(T0 - 400_000_000, side_end + 200_000_000, 20_000_000)
            ):
                q = [
                    (np.cos(0.013 * i + 2 * j) if r else np.sin(0.01 * i + j))
                    for j in range(7)
                ]
                put(meas, t + 3_000_000, kinds["JointState"](positions=q))
                put(cmd, t, kinds["JointState"](positions=[v + 0.01 for v in q]))
            hand = channel(f"{prefix}hand.command", "FingerCommand")
            for i, t in enumerate(
                range(T0 - 400_000_000, side_end + 200_000_000, 22_222_222)
            ):
                counts = [
                    ((i * (j + 2) + 500) if r else (i * (j + 1))) % 1001
                    + hand_count_offset
                    for j in range(6)
                ]
                put(hand, t, kinds["FingerCommand"](counts=counts))
            if tactile:
                tac = channel(f"{prefix}hand.tactile", "HandTactile")
                gap_here = side is None or side == tactile_gap_side
                tag = (tactile_side_tag or {}).get(side or "", side or "")
                for i, t in enumerate(range(T0 - 400_000_000, side_end, 95_000_000)):
                    frame = (t - T0) // PERIOD_NS
                    if (
                        gap_here
                        and tactile_gap
                        and tactile_gap[0] <= frame < tactile_gap[1]
                    ):
                        continue
                    put(
                        tac,
                        t + 5_000_000,
                        kinds["HandTactile"](
                            motor_valid=True,
                            sample_time_ns=t,
                            positions=[
                                ((i * 11 + 3 * j) if r else (i * 7 + j)) % 1000
                                for j in range(6)
                            ],
                            currents=[
                                ((i * 5 + j) if r else (i * 3 + j)) % 900 - 400
                                for j in range(6)
                            ],
                            side=tag,
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
        meta = {
            "task": "synthetic",
            "active_hand": "left",
            "recording_started_at_unix_ns": str(started),
        }
        if bimanual:
            meta |= BIMANUAL_META
        meta |= episode_meta or {}
        writer.add_metadata("episode", meta)
        writer.add_metadata("episode_outcome", {"outcome": outcome})
        spans = [] if reset_offset_ns is None else [list(reset_offset_ns)]
        for side in sides:
            name = "hand_reset" if side is None else f"hand_reset.{side}"
            here = spans if side is None or side == reset_side else []
            writer.add_metadata(
                name, {"events": str(len(here)), "spans_ns": json.dumps(here)}
            )
        writer.finish()
    disk: dict = {"outcome": outcome}
    if outcome_reset is not None:
        disk["hand_reset"] = outcome_reset
    (path / "outcome.json").write_text(json.dumps(disk))
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


# ------------------------------------------------------------------ bimanual

HAND_KEYS = ("current", "pos_err", "pos", "tip", "age", "motor_ok", "tip_ok")


@pytest.fixture(scope="module")
def bimanual(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # the right arm's streams stop 40 frames early: the AND must end the run there
    return _write(
        tmp_path_factory.mktemp("bim") / "ep0", bimanual=True, right_end_frames_early=40
    )


def test_bimanual_fills_both_sides_side_major(bimanual: Path) -> None:  # ruff:ignore[too-many-locals]
    ep = read_episode(bimanual)
    assert ep.bimanual
    assert ep.sides == ROBOT_SIDES
    assert ep.active_side == "both"
    with pytest.raises(AttributeError, match="bimanual"):
        _ = ep.measured_ts
    rows = build_rows(ep)
    c = rows.columns
    t = c["t_ns"]
    assert len(t) > 0
    assert c["side_valid"].shape == (len(t), 2)
    assert c["side_valid"].all()
    real = ~c["action.is_pad"]
    tk = t[:, None] + chunk_offsets_ns(100)[None]
    for s, side in enumerate(ROBOT_SIDES):
        arm = ep.arms[side]
        jm, _ = zoh(arm.measured_ts, t)
        jp, age = zoh(arm.hand_ts, t - PREV_NS)
        assert (age <= MAX_STALE_NS).all()
        np.testing.assert_array_equal(
            c["state"][:, s],
            np.concatenate([arm.measured[jm], arm.hand[jp]], axis=-1).astype(
                np.float32
            ),
        )
        jc, _ = zoh(arm.command_ts, tk.reshape(-1))
        jh, _ = zoh(arm.hand_ts, tk.reshape(-1))
        want = np.concatenate(
            [arm.command[jc].reshape(*tk.shape, 7), arm.hand[jh].reshape(*tk.shape, 6)],
            axis=-1,
        ).astype(np.float32)
        np.testing.assert_array_equal(c["action.chunk"][:, :, s][real], want[real])
    # the sides differ (a swap would not pass) and hand dims are counts/1000
    assert not np.array_equal(c["state"][:, 0], c["state"][:, 1])
    for name in ("state", "action.chunk"):
        h = c[name][..., 7:]
        assert h.min() >= 0
        assert h.max() <= 1
    # hand token columns are side-prefixed only
    for side in ROBOT_SIDES:
        for key in HAND_KEYS:
            assert f"hand.{side}.{key}" in c
    assert not any(k.startswith("hand.") and k.split(".")[1] in HAND_KEYS for k in c)
    frame = rows_to_frame(rows)
    assert frame["hand.right.current"].dtype.shape == (6,)  # ty:ignore[unresolved-attribute]
    assert frame["hand.left.tip"].dtype.shape == (10,)  # ty:ignore[unresolved-attribute]


def test_bimanual_validity_is_the_and_over_sides(bimanual: Path) -> None:
    ep = read_episode(bimanual)
    a = align(ep)
    assert a.j_measured is None
    left, right = a.per_side["left"].valid, a.per_side["right"].valid
    # the right arm stops early: left alone would keep frames the AND drops
    assert (left & ~right).sum() >= 30  # ruff:ignore[magic-value-comparison]
    cams = np.ones_like(left)
    for camera in ("side_left", "side_right"):
        cams &= a.side_dt[camera] <= SIDE_TOL_NS
    np.testing.assert_array_equal(a.valid, left & right & cams)
    rows = build_rows(ep)
    assert a.valid[rows.keep].all()
    assert rows.columns["t_ns"].max() <= ep.arms["right"].measured_ts[-1]


def test_bimanual_is_pad_is_the_union_and_both_sides_hold(bimanual: Path) -> None:  # ruff:ignore[too-many-locals]
    ep = read_episode(bimanual)
    rows = build_rows(ep)
    c = rows.columns
    t, pad = c["t_ns"], c["action.is_pad"]
    assert pad.shape == (len(t), 100)
    a = align(ep)
    ((start, stop),) = rows.stats["runs"]
    assert stop - start >= 100  # ruff:ignore[magic-value-comparison]
    t_end = ep.cam_ts["base"][stop - 1]
    tk = t[:, None] + chunk_offsets_ns(100)[None]
    stale = np.zeros(tk.shape, dtype=bool)
    for arm in ep.arms.values():
        _, age_c = zoh(arm.command_ts, tk.reshape(-1))
        _, age_h = zoh(arm.hand_ts, tk.reshape(-1))
        stale |= ((age_c > MAX_STALE_NS) | (age_h > MAX_STALE_NS)).reshape(tk.shape)
    want = np.logical_or.accumulate((tk > t_end) | stale, axis=1)
    want[:, 0] = False
    np.testing.assert_array_equal(pad, want)
    assert a.valid[rows.keep].all()
    # each side holds its value at the union's last real step
    i = len(pad) - 1
    k = int(pad[i].argmax())
    assert k > 0
    for s in range(2):
        np.testing.assert_array_equal(
            c["action.chunk"][i, k:, s],
            np.broadcast_to(c["action.chunk"][i, k - 1, s], (100 - k, 13)),
        )


def test_bimanual_hand_columns_are_the_shared_builder_per_side(tmp_path: Path) -> None:
    gap = _write(
        tmp_path / "gap",
        bimanual=True,
        tactile_gap=(100, 160),
        tactile_gap_side="right",
    )
    clean = _write(tmp_path / "clean", bimanual=True)
    ep = read_episode(gap)
    rows = build_rows(ep)
    c = rows.columns
    t = c["t_ns"]
    for side in ROBOT_SIDES:
        arm = ep.arms[side]
        tokens = hf.build_tokens(
            t,
            hf.motor_timeline(arm.tactile),
            hf.make_command_timeline(arm.hand_ts, arm.hand_counts),
            hf.tip_timeline(arm.tactile),
        )
        np.testing.assert_array_equal(c[f"hand.{side}.motor_ok"], tokens.motor_ok)
        for key in ("current", "pos_err", "pos", "tip"):
            np.testing.assert_array_equal(c[f"hand.{side}.{key}"], tokens.blocks[key])
        np.testing.assert_array_equal(
            c[f"hand.{side}.age"], tokens.blocks["hand_age"][:, 0]
        )
    # the gap is on the right hand only: rows are kept (no_hand frames), the left
    # hand is untouched, the right hand is refused through the gap
    base = build_rows(read_episode(clean)).columns
    np.testing.assert_array_equal(t, base["t_ns"])
    for key in HAND_KEYS:
        np.testing.assert_array_equal(c[f"hand.left.{key}"], base[f"hand.left.{key}"])
    assert (~c["hand.right.motor_ok"]).sum() > 40  # ruff:ignore[magic-value-comparison]
    assert (~c["hand.right.motor_ok"]).sum() > (~base["hand.right.motor_ok"]).sum()
    assert rows.stats["hand_ok_rows"]["right"] < rows.stats["hand_ok_rows"]["left"]


def test_bimanual_reset_spans_are_per_side_and_poison_both(tmp_path: Path) -> None:
    start = 500_000_000 + 150 * PERIOD_NS
    span = (start, start + PERIOD_NS)
    span_lo = T0 + 150 * PERIOD_NS - hf.RESET_PAD_BEFORE_NS
    span_hi = T0 + 151 * PERIOD_NS + hf.RESET_PAD_AFTER_NS

    def kept(mcap: Path) -> np.ndarray:
        return build_rows(read_episode(mcap), min_run=30).columns["t_ns"]

    def hit(t: np.ndarray) -> bool:
        return bool(((t >= span_lo) & (t <= span_hi)).any())

    # MCAP hand_reset.right: the row goes for BOTH sides
    assert not hit(
        kept(
            _write(
                tmp_path / "mcap",
                n_frames=700,
                bimanual=True,
                reset_offset_ns=span,
                reset_side="right",
            )
        )
    )
    # outcome.json hand_reset.by_side.left alone is honoured too
    by_side = {"by_side": {"left": {"events": 1, "spans_ns": [list(span)]}}}
    assert not hit(
        kept(
            _write(
                tmp_path / "disk", n_frames=700, bimanual=True, outcome_reset=by_side
            )
        )
    )
    # a bimanual take's top-level outcome hand_reset is a summary, never read
    summary = {"events": 1, "spans_ns": [list(span)]}
    assert hit(
        kept(
            _write(
                tmp_path / "summary", n_frames=700, bimanual=True, outcome_reset=summary
            )
        )
    )


def test_bimanual_refusals(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="active_hand 'left'"):
        read_episode(
            _write(
                tmp_path / "one",
                n_frames=60,
                bimanual=True,
                episode_meta={"active_hand": "left"},
            )
        )
    with pytest.raises(ValueError, match="counts outside"):
        read_episode(
            _write(tmp_path / "counts", n_frames=60, bimanual=True, hand_count_offset=1)
        )
    order = BIMANUAL_META["state_order"].split(",")
    swapped = ",".join(order[13:] + order[:13])
    with pytest.raises(ValueError, match="state_order"):
        read_episode(
            _write(
                tmp_path / "order",
                n_frames=60,
                bimanual=True,
                episode_meta={"state_order": swapped},
            )
        )
    with pytest.raises(ValueError, match="tagged side"):
        read_episode(
            _write(
                tmp_path / "tag",
                n_frames=60,
                bimanual=True,
                tactile_side_tag={"right": "left"},
            )
        )


def test_bimanual_windows_carry_both_sides(bimanual: Path) -> None:
    frame = NeroRobotReader()(bimanual)
    windows = NeroRobotWindowGrouper(clip_frames=8, episode_stride=7)(frame)
    assert len(windows) > 0
    assert windows["state"].dtype.shape == (8, 2, 13)  # ty:ignore[unresolved-attribute]
    assert windows["hand.right.motor_ok"].dtype.shape == (8,)  # ty:ignore[unresolved-attribute]
    assert windows["side_valid"].dtype.shape == (2,)  # ty:ignore[unresolved-attribute]
    assert np.stack(windows["side_valid"].to_numpy()).all()  # ty:ignore[no-matching-overload]


def test_layout_names_equal_the_nutron_cli_contract() -> None:
    assert len(layout_names()) == 26  # ruff:ignore[magic-value-comparison]
    assert layout_names()[:2] == ("left.joint1", "left.joint2")
    assert layout_names()[13] == "right.joint1"
    jetson = NUTRON_CLI / "runtime" / "jetson"
    if not (jetson / "policy_contract.py").exists():
        pytest.skip(f"no nutron-cli checkout at {NUTRON_CLI}")
    sys.path.insert(0, str(NUTRON_CLI / "runtime"))
    try:
        from jetson import policy_contract as pc  # ruff:ignore[import-outside-top-level]  # ty:ignore[unresolved-import]
    except ImportError as exc:
        pytest.skip(f"policy_contract not importable here: {exc}")
    if not hasattr(pc, "SIDES"):
        pytest.skip("nutron-cli checkout predates the bimanual layout")
    assert tuple(pc.SIDES) == ROBOT_SIDES
    assert layout_names() == tuple(pc.patch_axis_names(pc.SIDES))


# ------------------------------------- bimanual parity vs convert (real corpus)


def _bimanual_names() -> list[str]:
    names = sorted(p.name for p in BIMANUAL.glob("*") if (p / "data.mcap").exists())
    if not names:
        return ["<none>"]
    # a handful by default (both left- and right-dominant takes); all 85 on demand
    return names if os.environ.get("NUTRON_BIMANUAL_PARITY_ALL") else names[::17]


@pytest.mark.parametrize("name", _bimanual_names())
def test_bimanual_align_parity_with_convert(name: str) -> None:
    """Contract rule 8: `align().valid` and the per-side indices ARE convert's.

    Only the alignment is pinned. The post-align filters differ by design
    (patch: poison/duplicate/max_pad; ACT: runs_of(valid) only), so the row sets
    are not compared -- but every kept row's state / action[0] must equal
    convert's vectors at that frame.
    """
    if name == "<none>":
        pytest.skip(f"no bimanual episodes under {BIMANUAL}")
    convert = _convert()
    if not hasattr(convert, "align_side"):
        pytest.skip("nutron-cli convert.py predates the bimanual reader")
    ref = convert.read_mcap(BIMANUAL / name)
    a_ref = convert.align(ref)
    ep = read_episode(BIMANUAL / name / "data.mcap")
    a = align(ep)
    np.testing.assert_array_equal(a.valid, a_ref.valid)
    for side in ROBOT_SIDES:
        mine, theirs = a.per_side[side], a_ref.per_side[side]
        np.testing.assert_array_equal(mine.valid, theirs.valid)
        np.testing.assert_array_equal(mine.j_hand_prev, theirs.j_hand_prev)
        np.testing.assert_array_equal(mine.j_measured, theirs.j_measured)
        np.testing.assert_array_equal(mine.j_command, theirs.j_command)
        np.testing.assert_array_equal(mine.j_hand, theirs.j_hand)
    rows = build_rows(ep)
    i = rows.keep
    assert a_ref.valid[i].all()
    # convert's runs_of(valid) frames are a superset of rbyte's kept rows
    act = np.zeros(len(a_ref.valid), dtype=bool)
    for start, stop in runs_of(a_ref.valid, 100):
        act[start:stop] = True
    assert act[i].all()
    want = [convert.frame_vectors(ref, a_ref, int(k)) for k in i]
    np.testing.assert_array_equal(
        rows.columns["state"].reshape(len(i), 26), np.stack([w[0] for w in want])
    )
    np.testing.assert_array_equal(
        rows.columns["action.chunk"][:, 0].reshape(len(i), 26),
        np.stack([w[1] for w in want]),
    )
    for camera in ("base", "side_left", "side_right"):
        pick = i if camera == "base" else a_ref.side_pick[camera][i]
        np.testing.assert_array_equal(
            rows.columns[f"frame_index.{camera}"], ref.cams[camera].idx[pick]
        )


def test_bimanual_corpus_probe() -> None:
    """Read-only probe of the whole 2026-10-07 corpus (opt-in: ~1 min on NFS)."""
    if not os.environ.get("NUTRON_BIMANUAL_PROBE"):
        pytest.skip("set NUTRON_BIMANUAL_PROBE=1 to ingest the whole corpus")
    names = sorted(p for p in BIMANUAL.glob("*") if (p / "data.mcap").exists())
    if not names:
        pytest.skip(f"no bimanual episodes under {BIMANUAL}")
    grouper = NeroRobotWindowGrouper(clip_frames=32, frame_stride=3, episode_stride=7)
    total = {"frames": 0, "valid": 0, "rows": 0, "windows": 0}
    for p in names:
        rows = build_rows(read_episode(p / "data.mcap"))
        for key in ("frames", "valid", "rows"):
            total[key] += rows.stats[key]
        total["windows"] += len(grouper(rows_to_frame(rows)))
        assert rows.stats["rows"] > 0, p.name
    assert len(names) == 85  # ruff:ignore[magic-value-comparison]
    # the frozen corpus: ACT's valid frame count, ~19.6k patch rows, 1,687 windows
    assert total == {"frames": 29285, "valid": 23942, "rows": 19592, "windows": 1687}
