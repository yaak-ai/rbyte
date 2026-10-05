"""nero robot-native episodes -> 30 Hz rows for the patch policy (P4, P6, P10).

The ROBOT recordings (nutron-cli `record.py`: `robot.measured.q`/`robot.command.q`
at ~50 Hz, `robot.hand.command` at ~45-49 Hz, optional `robot.hand.tactile`, three
OAK cameras at 30 fps) are a different source from the glove recordings that
`NeroArmsReader` reads. Their clock is the MCAP `publish_time` (the
host master clock; for cameras the mid-exposure time), NOT the glove device clock,
and the rules that turn them into training rows already exist in nutron-cli's
`runtime/training/convert.py`, which builds the ACT dataset. This module PORTS
those rules -- rbyte cannot import nutron-cli -- and `tests/test_nero_robot.py`
pins the port against `convert.align` on a pulled episode when one is present.

Ported verbatim from convert.py (keep in sync):

* `t` = the base camera's `publish_time` (int64 ns);
* arm measured/command and hand command are zero-order held at `t` with age
  <= 100 ms AND `t <= last sample` (`convert.zoh`, `MAX_STALE_NS`);
* `hand_prev` = hand command ZOH at `t - 1/30 s` with age <= 100 ms (`PREV_NS`);
* side cameras: nearest frame within 50 ms (`SIDE_TOL_NS`);
* valid spans shorter than 100 frames are dropped (`MIN_RUN`).

What this builder adds on top (the patch family, contract v3):

* **State** per side = measured q (7) + hand_prev / 1000 (6) -- convert's
  `observation.state`.
* **Action chunk** per side = `chunk[k]` = command q (7) + hand command / 1000
  (6), zero-order held at `t + k/30 s` for `k = 0..chunk_size-1`. `chunk[0]` is
  bit-identical to convert's `action` (`chunk_t0_offset_steps = 0`).
* **Hold-padding.** A step whose time runs past the end of the row's valid run
  (episode end, a held/stale gap, or a reset span) holds the last real command
  and is flagged in `action.is_pad`; rows with more than `max_pad_steps` padded
  steps are dropped. Without padding a 100-step chunk would throw away the last
  3.3 s of every run -- the release/retreat phase.
* **Hand token blocks**, built by the vendored nutron-cli `hand_features`
  (`build_tokens`: newest sample only + age + validity). Never NULL: a refused
  frame is zeros with `hand.motor_ok = False`, and the MODEL substitutes its
  learned `no_hand` token. Only recorder/self-detected reset spans
  (`reset_poison`) exclude rows, exactly as in ACT.
* **`grid_index`**, a gap-aware 30 Hz frame counter (a dropped camera frame
  advances it by 2). The 10 Hz windowing (`NeroRobotWindowGrouper`) requires a
  complete run of `grid_index` values, so a window can never silently span a
  dropped frame, a stale-state gap or a reset span.

Emitted columns (one row per kept base frame, `S = 2` sides, `A = 13`):

| column                       | dtype                         |
|------------------------------|-------------------------------|
| `frame_index.{camera}`       | `Int32` (mp4 ordinal)         |
| `grid_index`                 | `Int64`                       |
| `t_ns`                       | `Int64`                       |
| `state`                      | `Array(Float32, (2, 13))`     |
| `action.chunk`               | `Array(Float32, (C, 2, 13))`  |
| `action.is_pad`              | `Array(Boolean, C)`           |
| `side_valid`                 | `Array(Boolean, 2)`           |
| `hand.current/pos_err/pos`   | `Array(Float32, 6)`           |
| `hand.tip`                   | `Array(Float32, 10)`          |
| `hand.age`                   | `Float32` (age / 250 ms, clip 2) |
| `hand.motor_ok` / `hand.tip_ok` | `Boolean`                  |
| `camera_cond`                | `Array(Float32, (3, 13))`     |
| `camera_cond.placeholder`    | `Boolean`                     |
"""

import json
import operator
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import starmap
from os import PathLike
from pathlib import Path
from typing import Any, Final, final

import numpy as np
import numpy.typing as npt
import polars as pl
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from pydantic import NonNegativeInt, PositiveInt, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars

from rbyte.samples.nero._vendor import hand_features as hf

logger = get_logger(__name__)

__all__ = [
    "ACTION_DIM",
    "AXIS_NAMES",
    "FINGER_NAMES",
    "JOINT_NAMES",
    "ROBOT_CAMERAS",
    "ROBOT_SIDES",
    "NeroRobotReader",
    "NeroRobotWindowGrouper",
    "RobotEpisode",
    "align",
    "build_rows",
    "chunk_offsets_ns",
    "read_episode",
    "runs_of",
]

# ------------------------------------------------- convert.py port (keep in sync)

FPS: Final = 30
MAX_STALE_NS: Final = 100_000_000
SIDE_TOL_NS: Final = 50_000_000
PREV_NS: Final = 1_000_000_000 // FPS
MIN_RUN: Final = 100

T_MEASURED: Final = "robot.measured.q"
T_COMMAND: Final = "robot.command.q"
T_HAND: Final = "robot.hand.command"
T_TACTILE: Final = "robot.hand.tactile"

ROBOT_CAMERAS: Final = ("base", "side_left", "side_right")
ROBOT_SIDES: Final = ("left", "right")
JOINT_NAMES: Final = tuple(f"joint{i}" for i in range(1, 8))
FINGER_NAMES: Final = ("thumb_flex", "thumb_aux", "index", "middle", "ring", "pinky")
AXIS_NAMES: Final = JOINT_NAMES + FINGER_NAMES
ACTION_DIM: Final = len(AXIS_NAMES)
HAND_SCALE: Final = 1000.0

#: nutron_paths.CAMERA_MXIDS: the rig's immutable camera serials. Early
#: recordings swapped the two side NAMES; the serials say which is which.
CAMERA_MXIDS: Final = {
    "194430100104372F00": "base",
    "19443010A1A4FC5A00": "side_left",
    "194430103106645A00": "side_right",
}

_HAND_RESET_KEY: Final = "hand_reset"
_NS_PER_S: Final = 1_000_000_000


def zoh(
    src_ts: npt.NDArray[np.int64], t: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """`convert.zoh`: zero-order-hold index into `src_ts` and the sample's age."""
    j = np.searchsorted(src_ts, t, side="right") - 1
    age = np.where(j >= 0, t - src_ts[np.maximum(j, 0)], np.iinfo(np.int64).max)
    return np.maximum(j, 0), age


def nearest(
    cam_ts: npt.NDArray[np.int64], t: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """`convert.nearest`: nearest-neighbour index into `cam_ts`, plus |dt|."""
    r = np.searchsorted(cam_ts, t, side="left")
    lo = np.clip(r - 1, 0, len(cam_ts) - 1)
    hi = np.clip(r, 0, len(cam_ts) - 1)
    dlo = np.abs(t - cam_ts[lo])
    dhi = np.abs(t - cam_ts[hi])
    take_lo = dlo <= dhi
    return np.where(take_lo, lo, hi), np.where(take_lo, dlo, dhi)


def runs_of(valid: npt.NDArray[np.bool_], min_run: int) -> list[tuple[int, int]]:
    """`convert.runs_of`: maximal `[start, stop)` spans of True, length >= min_run."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(valid)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        if j - i >= min_run:
            out.append((i, j))
        i = j
    return out


def chunk_offsets_ns(chunk_size: int) -> npt.NDArray[np.int64]:
    """`k/30 s` for `k = 0..chunk_size-1`, exact integer ns (k * 1e9 // 30)."""
    return (np.arange(chunk_size, dtype=np.int64) * _NS_PER_S) // FPS


# -------------------------------------------------------------------- reading


@dataclass
class RobotEpisode:
    """One robot recording, every stream on the MCAP `publish_time` clock."""

    name: str
    measured_ts: npt.NDArray[np.int64]
    measured: npt.NDArray[np.float64]  # (N, 7)
    command_ts: npt.NDArray[np.int64]
    command: npt.NDArray[np.float64]  # (N, 7)
    hand_ts: npt.NDArray[np.int64]
    hand_counts: npt.NDArray[np.int64]  # (N, 6) raw 0..1000
    cam_ts: dict[str, npt.NDArray[np.int64]]
    cam_idx: dict[str, npt.NDArray[np.int64]]
    tactile: list[Any]
    active_side: str = "left"
    recording_started_ns: int | None = None
    #: recorder reset spans as OFFSETS from recording start (MCAP U outcome.json)
    hand_reset_spans_offset_ns: list[list[int]] | None = None

    @property
    def hand(self) -> npt.NDArray[np.float64]:
        """`robot.hand.command / 1000` -- convert's `ep.hand.val`."""
        return self.hand_counts.astype(np.float64) / HAND_SCALE


def _sorted_by_publish(
    rows: list[tuple[int, Any]],
) -> tuple[npt.NDArray[np.int64], list[Any]]:
    """`convert._sorted_by_publish`: a STABLE sort on publish_time."""
    rows.sort(key=operator.itemgetter(0))
    ts = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    return ts, [r[1] for r in rows]


def _camera_sources(metadata: Mapping[str, Mapping[str, str]]) -> None:
    """Refuse a recording whose stored camera names are not the physical ones.

    nutron-cli resolves swapped side names through the camera serials; rbyte's
    dataset config addresses `<camera>.mp4` by NAME, so a swapped recording
    would silently feed the wrong view. Identity or no metadata passes.

    Raises:
        ValueError: on partial, unknown or swapped camera serial metadata.
    """
    rows = {name: metadata.get(f"camera.{name}", {}) for name in ROBOT_CAMERAS}
    if not any(rows.values()):
        return
    if not all(rows.values()):
        msg = "recording has incomplete camera serial metadata"
        raise ValueError(msg)
    for stored, row in rows.items():
        physical = CAMERA_MXIDS.get(str(row.get("mxid") or ""))
        if physical is None:
            msg = f"unknown camera serial {row.get('mxid')!r}"
            raise ValueError(msg)
        if physical != stored:
            msg = (
                f"stored camera {stored!r} is physically {physical!r} (swapped side "
                "names); rbyte addresses mp4s by name -- remap before ingesting"
            )
            raise ValueError(msg)


def _reset_spans_offset(
    metadata: Mapping[str, Mapping[str, str]], outcome: Mapping[str, Any]
) -> list[list[int]]:
    """`convert.hand_reset_spans`, but the UNION of both copies (brief: MCAP U disk)."""
    spans: list[list[int]] = []
    for source in (
        metadata.get(_HAND_RESET_KEY, {}),
        outcome.get(_HAND_RESET_KEY) or {},
    ):
        value = source.get("spans_ns") or "[]"
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = []
        if isinstance(value, list):
            spans += [[int(s), int(e)] for s, e in value]
    return spans


def read_episode(  # ruff:ignore[complex-structure, too-many-statements, too-many-locals]
    path: PathLike[str] | str,
) -> RobotEpisode:
    """Read `data.mcap` (+ the sibling `outcome.json` if present).

    Raises:
        ValueError: on a missing stream or a malformed message.
    """
    path = Path(path)
    topics = [T_MEASURED, T_COMMAND, T_HAND, T_TACTILE] + [
        f"observation.images.{c}" for c in ROBOT_CAMERAS
    ]
    raw: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    with path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        metadata = {r.name: dict(r.metadata) for r in reader.iter_metadata()}
        f.seek(0)
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=topics
        ):
            raw[channel.topic].append((message.publish_time, decoded))

    _camera_sources(metadata)

    def joints(topic: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        ts, rows = _sorted_by_publish(raw[topic])
        if not len(ts):
            msg = f"{path}: no {topic} messages"
            raise ValueError(msg)
        val = np.asarray([list(r.positions) for r in rows], dtype=np.float64)
        if val.shape[1:] != (7,):
            msg = f"{path}: {topic} positions are not 7-wide: {val.shape}"
            raise ValueError(msg)
        return ts, val

    measured_ts, measured = joints(T_MEASURED)
    command_ts, command = joints(T_COMMAND)

    hand_ts, hand_rows = _sorted_by_publish(raw[T_HAND])
    if not len(hand_ts):
        msg = f"{path}: no {T_HAND} messages"
        raise ValueError(msg)
    hand_counts = np.asarray([list(r.counts) for r in hand_rows], dtype=np.int64)
    if hand_counts.shape[1:] != (6,):
        msg = f"{path}: {T_HAND} counts are not 6-wide"
        raise ValueError(msg)

    _, tactile = _sorted_by_publish(raw[T_TACTILE])

    cam_ts: dict[str, npt.NDArray[np.int64]] = {}
    cam_idx: dict[str, npt.NDArray[np.int64]] = {}
    for camera in ROBOT_CAMERAS:
        ts, rows = _sorted_by_publish(raw[f"observation.images.{camera}"])
        if not len(ts):
            msg = f"{path}: no observation.images.{camera} messages"
            raise ValueError(msg)
        # proto3: the first row's frame_index is absent on the wire (== 0)
        idx = np.fromiter(
            (int(r.frame_index) for r in rows), dtype=np.int64, count=len(rows)
        )
        if not np.all(np.diff(idx) > 0):
            msg = f"{path}: {camera} frame_index not increasing"
            raise ValueError(msg)
        cam_ts[camera], cam_idx[camera] = ts, idx

    episode = metadata.get("episode", {})
    started = episode.get("recording_started_at_unix_ns")
    outcome_path = path.parent / "outcome.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    # convert.py's corpus rule, ported: a take trains only if BOTH copies of its
    # outcome say success -- the MCAP record (cannot be edited after the fact) and
    # outcome.json (what a web-UI relabel rewrites). Anything else is refused by
    # name rather than silently ingested as a demonstration.
    recorded = metadata.get("episode_outcome", {}).get("outcome")
    if recorded != "success":
        msg = f"{path}: mcap outcome is {recorded!r}, not success"
        raise ValueError(msg)
    if outcome.get("outcome") != "success":
        msg = f"{path}: outcome.json is {outcome.get('outcome')!r}, not success"
        raise ValueError(msg)
    active = str(episode.get("active_hand") or "left")
    if active not in ROBOT_SIDES:
        msg = f"{path}: active_hand {active!r} not in {ROBOT_SIDES}"
        raise ValueError(msg)

    return RobotEpisode(
        name=path.parent.name,
        measured_ts=measured_ts,
        measured=measured,
        command_ts=command_ts,
        command=command,
        hand_ts=hand_ts,
        hand_counts=hand_counts,
        cam_ts=cam_ts,
        cam_idx=cam_idx,
        tactile=tactile,
        active_side=active,
        recording_started_ns=int(started) if started else None,
        hand_reset_spans_offset_ns=_reset_spans_offset(metadata, outcome),
    )


# ------------------------------------------------------------------ alignment


@dataclass
class Aligned:
    """`convert.Aligned`: per-base-frame alignment of one episode."""

    valid: npt.NDArray[np.bool_]
    j_measured: npt.NDArray[np.intp]
    j_command: npt.NDArray[np.intp]
    j_hand: npt.NDArray[np.intp]
    j_hand_prev: npt.NDArray[np.intp]
    side_pick: dict[str, npt.NDArray[np.intp]]
    side_dt: dict[str, npt.NDArray[np.int64]]


def align(ep: RobotEpisode) -> Aligned:
    """`convert.align`, line for line."""
    t = ep.cam_ts["base"]
    valid = np.ones(len(t), dtype=bool)

    js = {}
    for name, ts in (
        ("measured", ep.measured_ts),
        ("command", ep.command_ts),
        ("hand", ep.hand_ts),
    ):
        j, age = zoh(ts, t)
        valid &= (age <= MAX_STALE_NS) & (t <= ts[-1])
        js[name] = j

    j_hand_prev, hand_prev_age = zoh(ep.hand_ts, t - PREV_NS)
    valid &= hand_prev_age <= MAX_STALE_NS

    side_pick, side_dt = {}, {}
    for camera in ("side_left", "side_right"):
        pick, dt = nearest(ep.cam_ts[camera], t)
        valid &= dt <= SIDE_TOL_NS
        side_pick[camera], side_dt[camera] = pick, dt

    return Aligned(
        valid=valid,
        j_measured=js["measured"],
        j_command=js["command"],
        j_hand=js["hand"],
        j_hand_prev=j_hand_prev,
        side_pick=side_pick,
        side_dt=side_dt,
    )


def grid_index(t_ns: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Gap-aware 30 Hz counter: +1 per nominal frame period, +2 over a dropped frame.

    Built from successive differences (not `(t - t0) * 30`) so clock-rate error
    cannot accumulate into spurious gaps over a long take. A duplicate frame
    (difference rounding to 0) keeps its predecessor's index and is reported by
    `duplicate_grid`.
    """
    if len(t_ns) == 0:
        return np.zeros(0, dtype=np.int64)
    period = _NS_PER_S / FPS
    steps = np.rint(np.diff(t_ns).astype(np.float64) / period).astype(np.int64)
    return np.concatenate([[0], np.cumsum(np.maximum(steps, 0))]).astype(np.int64)


def duplicate_grid(grid: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    """Rows whose `grid_index` repeats the previous row's."""
    out = np.zeros(len(grid), dtype=bool)
    out[1:] = np.diff(grid) == 0
    return out


# ----------------------------------------------------------------------- rows


@dataclass
class Rows:
    """Every emitted column for one episode, before the DataFrame wrap."""

    keep: npt.NDArray[np.intp]  # base-frame indices that became rows
    columns: dict[str, npt.NDArray[Any]]
    stats: dict[str, Any]


def _reset_spans(ep: RobotEpisode, motor: hf.MotorTimeline) -> npt.NDArray[np.int64]:
    """Recorder spans (MCAP U outcome.json) UNION self-detected, absolute ns."""
    spans = [hf.reset_spans_from_timeline(motor)]
    if ep.hand_reset_spans_offset_ns and ep.recording_started_ns is not None:
        spans.append(
            hf.reset_spans_from_record(
                ep.hand_reset_spans_offset_ns, ep.recording_started_ns
            )
        )
    elif ep.hand_reset_spans_offset_ns:
        logger.warning(
            "hand_reset spans recorded without recording_started_at_unix_ns; "
            "using self-detection only",
            episode=ep.name,
        )
    return np.concatenate(spans, axis=0).astype(np.int64).reshape(-1, 2)


def build_rows(  # ruff:ignore[too-many-locals, too-many-statements]
    ep: RobotEpisode,
    *,
    chunk_size: int = 100,
    max_pad_steps: int | None = None,
    min_run: int = MIN_RUN,
    camera_cond: npt.NDArray[np.float32] | None = None,
) -> Rows:
    """All row columns for one episode (see the module docstring).

    Raises:
        ValueError: if `chunk_size` < 1.
    """
    if chunk_size < 1:
        msg = "chunk_size must be >= 1"
        raise ValueError(msg)
    max_pad = chunk_size // 2 if max_pad_steps is None else max_pad_steps
    t = ep.cam_ts["base"]
    n = len(t)
    a = align(ep)

    motor = hf.motor_timeline(ep.tactile)
    tips = hf.tip_timeline(ep.tactile)
    commands = hf.make_command_timeline(ep.hand_ts, ep.hand_counts)
    tokens = hf.build_tokens(t, motor, commands, tips)
    spans = _reset_spans(ep, motor)
    poison = hf.reset_poison(t, spans)

    grid = grid_index(t)
    duplicate = duplicate_grid(grid)
    usable = a.valid & ~poison & ~duplicate
    runs = runs_of(usable, min_run)

    offsets = chunk_offsets_ns(chunk_size)
    hand = ep.hand
    side = ROBOT_SIDES.index(ep.active_side)

    keep: list[int] = []
    chunks: list[npt.NDArray[np.float64]] = []
    pads: list[npt.NDArray[np.bool_]] = []
    dropped_pad = 0
    for start, stop in runs:
        rows = np.arange(start, stop)
        t_end = t[stop - 1]
        tk = t[rows, None] + offsets[None, :]  # (R, C)
        jc, age_c = zoh(ep.command_ts, tk.reshape(-1))
        jh, age_h = zoh(ep.hand_ts, tk.reshape(-1))
        stale = ((age_c > MAX_STALE_NS) | (age_h > MAX_STALE_NS)).reshape(tk.shape)
        # a pad is a SUFFIX: the first padded step and everything after it
        pad = np.logical_or.accumulate((tk > t_end) | stale, axis=1)
        pad[:, 0] = False  # k = 0 is t itself, a valid frame by construction
        chunk = np.concatenate(
            [ep.command[jc].reshape(*tk.shape, 7), hand[jh].reshape(*tk.shape, 6)],
            axis=-1,
        )
        # hold the last real step through the padded tail
        last = np.maximum((~pad).sum(axis=1) - 1, 0)
        held = chunk[np.arange(len(rows)), last]  # (R, 13)
        chunk = np.where(pad[..., None], held[:, None, :], chunk)
        ok = pad.sum(axis=1) <= max_pad
        dropped_pad += int((~ok).sum())
        keep += rows[ok].tolist()
        chunks.append(chunk[ok])
        pads.append(pad[ok])

    keep_idx = np.asarray(keep, dtype=np.intp)
    r = len(keep_idx)
    chunk_all = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.zeros((0, chunk_size, ACTION_DIM))
    )
    pad_all = (
        np.concatenate(pads, axis=0) if pads else np.zeros((0, chunk_size), dtype=bool)
    )

    state = np.zeros((r, len(ROBOT_SIDES), ACTION_DIM), dtype=np.float32)
    state[:, side] = np.concatenate(
        [ep.measured[a.j_measured[keep_idx]], hand[a.j_hand_prev[keep_idx]]], axis=-1
    ).astype(np.float32)
    action = np.zeros((r, chunk_size, len(ROBOT_SIDES), ACTION_DIM), dtype=np.float32)
    action[:, :, side] = chunk_all.astype(np.float32)
    side_valid = np.zeros((r, len(ROBOT_SIDES)), dtype=bool)
    side_valid[:, side] = True

    cond = (
        np.zeros((len(ROBOT_CAMERAS), 13), dtype=np.float32)
        if camera_cond is None
        else np.asarray(camera_cond, dtype=np.float32).reshape(len(ROBOT_CAMERAS), 13)
    )

    columns: dict[str, npt.NDArray[Any]] = {
        "frame_index.base": ep.cam_idx["base"][keep_idx].astype(np.int32),
        **{
            f"frame_index.{c}": ep.cam_idx[c][a.side_pick[c][keep_idx]].astype(np.int32)
            for c in ("side_left", "side_right")
        },
        "grid_index": grid[keep_idx],
        "t_ns": t[keep_idx],
        "state": state,
        "action.chunk": action,
        "action.is_pad": pad_all,
        "side_valid": side_valid,
        "hand.current": tokens.blocks["current"][keep_idx],
        "hand.pos_err": tokens.blocks["pos_err"][keep_idx],
        "hand.pos": tokens.blocks["pos"][keep_idx],
        "hand.tip": tokens.blocks["tip"][keep_idx],
        "hand.age": tokens.blocks["hand_age"][keep_idx, 0],
        "hand.motor_ok": tokens.motor_ok[keep_idx],
        "hand.tip_ok": tokens.tip_ok[keep_idx],
        "camera_cond": np.broadcast_to(cond, (r, *cond.shape)).copy(),
        "camera_cond.placeholder": np.full(r, camera_cond is None),
    }
    stats = {
        "frames": n,
        "valid": int(a.valid.sum()),
        "reset_poison": int(poison.sum()),
        "duplicate_grid": int(duplicate.sum()),
        "runs": runs,
        "rows": r,
        "dropped_pad": dropped_pad,
        "padded_rows": int(pad_all.any(axis=1).sum()),
        "hand_ok_rows": int(tokens.motor_ok[keep_idx].sum()),
        "tactile_rows": len(ep.tactile),
        "motor_rows": len(motor),
    }
    return Rows(keep=keep_idx, columns=columns, stats=stats)


def _series(name: str, values: npt.NDArray[Any]) -> pl.Series:
    if np.issubdtype(values.dtype, np.floating):
        inner: pl.DataType = pl.Float32()
        values = values.astype(np.float32)
    elif values.dtype == np.bool_:
        inner = pl.Boolean()
    elif values.dtype == np.int32:
        inner = pl.Int32()
    else:
        inner = pl.Int64()
        values = values.astype(np.int64)
    dtype = inner
    for size in reversed(values.shape[1:]):
        dtype = pl.Array(dtype, size)
    return pl.Series(name, values, dtype=dtype)


def rows_to_frame(rows: Rows) -> pl.DataFrame:
    return pl.DataFrame(list(starmap(_series, rows.columns.items())))


@final
class NeroRobotReader:
    """`data.mcap` of one robot recording -> 30 Hz rows (see module docstring)."""

    @validate_call
    def __init__(
        self,
        *,
        chunk_size: PositiveInt = 100,
        max_pad_steps: NonNegativeInt | None = None,
        min_run: PositiveInt = MIN_RUN,
        camera_cond: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._max_pad_steps = max_pad_steps
        self._min_run = min_run
        self._camera_cond = (
            None if camera_cond is None else np.asarray(camera_cond, dtype=np.float32)
        )

    def __call__(self, path: PathLike[str] | str) -> pl.DataFrame:
        with bound_contextvars(path=str(path)):
            rows = build_rows(
                read_episode(path),
                chunk_size=self._chunk_size,
                max_pad_steps=self._max_pad_steps,
                min_run=self._min_run,
                camera_cond=self._camera_cond,
            )
            logger.debug("built robot rows", **rows.stats)
            return rows_to_frame(rows)


# ------------------------------------------------------------------ windowing


@final
class NeroRobotWindowGrouper:
    """30 Hz rows -> `T`-frame samples on a `frame_stride` grid (P2: 10 Hz).

    A window starting at grid index `g` takes rows `g, g+s, ..., g+(T-1)s` and is
    emitted ONLY if every grid index `g .. g+(T-1)s` is present. That rules out
    the `gather_every` trap -- gathering every s-th ROW of a window with a missing
    row yields non-uniform 66/133 ms spacing the KV/RoPE model would read as
    100 ms -- by construction, and it also excludes windows across stale-state
    gaps, dropped camera frames and reset spans (their rows were never emitted).

    Window starts are `g0 + episode_offset + m * episode_stride`. Pick
    `episode_stride` coprime to `frame_stride` so the starts cycle through all
    `frame_stride` phase offsets (all 30 Hz frames are used as 10 Hz data).

    Columns in `constant_columns` are per-episode constants collapsed to the first
    frame's value; every other column becomes `Array(..., (T, *shape))`.
    """

    @validate_call
    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        clip_frames: PositiveInt,
        frame_stride: PositiveInt = 3,
        episode_stride: PositiveInt = 1,
        episode_offset: NonNegativeInt = 0,
        index_column: str = "grid_index",
        constant_columns: Sequence[str] = (
            "side_valid",
            "camera_cond",
            "camera_cond.placeholder",
        ),
    ) -> None:
        self._t = clip_frames
        self._stride = frame_stride
        self._episode_stride = episode_stride
        self._episode_offset = episode_offset
        self._index = index_column
        self._constant = tuple(constant_columns)

    def starts(self, grid: npt.NDArray[np.int64]) -> npt.NDArray[np.intp]:
        """Row positions of every emitted window's first frame (grid sorted)."""
        if len(grid) == 0:
            return np.zeros(0, dtype=np.intp)
        span = (self._t - 1) * self._stride
        position = {int(g): i for i, g in enumerate(grid.tolist())}
        out = []
        for g in range(
            int(grid[0]) + self._episode_offset,
            int(grid[-1]) - span + 1,
            self._episode_stride,
        ):
            i = position.get(g)
            # rows are unique and sorted on grid, so "g + span sits exactly span
            # rows later" <=> every grid index in [g, g + span] is present
            if i is not None and i + span < len(grid) and grid[i + span] == g + span:
                out.append(i)
        return np.asarray(out, dtype=np.intp)

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        frame = input.sort(self._index)
        grid = frame[self._index].to_numpy().astype(np.int64)
        if len(np.unique(grid)) != len(grid):
            msg = f"{self._index} is not unique"
            raise ValueError(msg)
        starts = self.starts(grid)
        gather = starts[:, None] + np.arange(self._t)[None, :] * self._stride  # (W, T)
        columns: list[pl.Series] = []
        for name in frame.columns:
            series = frame[name]
            if name in self._constant:
                columns.append(series.gather(starts))
                continue
            values = series.to_numpy()
            out = values[gather.reshape(-1)].reshape(
                len(starts), self._t, *values.shape[1:]
            )
            dtype = series.dtype
            for size in reversed((self._t,)):
                dtype = pl.Array(dtype, size)
            columns.append(pl.Series(name, out, dtype=dtype))
        result = pl.DataFrame(columns)
        logger.debug("windowed", windows=len(result), rows=len(frame))
        return result
