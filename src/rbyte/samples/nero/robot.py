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

BIMANUAL (nutron-cli contract layout "nero-bimanual-26"). A recording whose MCAP
`episode` metadata says `capture_mode = bus_bimanual` (`active_hand = both`) is
read from the per-side topics `robot.{left,right}.{measured.q, command.q,
hand.command, hand.tactile}` -- selected from the metadata, never guessed from
which topics exist -- and every rule above runs ONCE PER SIDE:

* **Validity** (contract rule 8, the same rule as `convert.align`): a frame is
  valid iff, for EVERY side, measured/command/hand command are fresh within the
  100 ms ZOH and `t <=` that side's last sample, and that side's `hand_prev` is
  fresh; plus the side-camera match. `align(ep).per_side[side]` holds each arm's
  own validity and indices (what the parity test pins against convert).
* **State / chunk** fill both sides (side-major, `[left 13 | right 13]`),
  `side_valid = [True, True]`. Hand dims are counts/1000 and asserted in [0, 1].
* **`action.is_pad`** stays `(C,)`: the per-step UNION over sides (a step is
  padded if it runs past the run end or EITHER side's command is stale), and
  both sides hold their value at the union's last real step.
* **Reset spans** are read per side -- MCAP `hand_reset.{side}` U outcome.json
  `hand_reset.by_side.{side}` (the top-level outcome `hand_reset` of a bimanual
  take is an events-only summary and is NOT read) -- plus per-side
  self-detection; `reset_poison` is the union over sides.
* **Hand token blocks** are the UNCHANGED vendored `hf.build_tokens` called once
  per side on that side's tactile + command, emitted as side-prefixed columns
  `hand.{left,right}.{current,pos_err,pos,tip,age,motor_ok,tip_ok}` (same
  dtypes as above). No unsided `hand.*` columns are emitted for a bimanual take.
* The recorder's `state_order` / `action_order` metadata must equal the
  canonical layout; its `state_hand_source` is ignored (the state hand part is
  always `hand_prev`).

POST-ALIGN FILTERS ARE PATCH-ONLY, BY DESIGN. After `align`, this builder keeps
`runs_of(valid & ~reset_poison & ~duplicate_grid, MIN_RUN)` and then drops rows
whose chunk pads more than `max_pad_steps`. nutron-cli's ACT `convert.py` keeps
`runs_of(valid, MIN_RUN)` and carries poison as a column (it becomes chunk
padding there). The two families therefore train on different row sets (about
19.6k patch rows vs 23.9k ACT frames on the 2026-10-07 bimanual corpus); only
`align().valid` and the per-side `hand_prev` indices are shared and pinned.

A single-arm recording (no `bus_bimanual` capture mode) takes the legacy path
and its output is unchanged.
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
    "RobotArm",
    "RobotEpisode",
    "align",
    "build_rows",
    "chunk_offsets_ns",
    "layout_names",
    "read_episode",
    "runs_of",
    "side_topics",
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
_HAND_RESET_BY_SIDE: Final = "by_side"
#: MCAP `episode.capture_mode` of a two-arm take (nutron-cli convert.py)
BIMANUAL_CAPTURE_MODE: Final = "bus_bimanual"
BIMANUAL_ACTIVE_HAND: Final = "both"
HAND_COUNT_MAX: Final = 1000
#: the recorder's own units for the 26-d order it declares
_RECORDER_ARM_UNITS: Final = "radians"
_RECORDER_HAND_UNITS: Final = "revo2_counts_0_to_1000"


def side_topics(side: str | None) -> dict[str, str]:
    """The four robot streams of one arm: `robot.*` (single-arm) or `robot.{side}.*`."""
    prefix = "robot." if side is None else f"robot.{side}."
    return {
        "measured": f"{prefix}measured.q",
        "command": f"{prefix}command.q",
        "hand": f"{prefix}hand.command",
        "tactile": f"{prefix}hand.tactile",
    }


def layout_names(sides: Sequence[str] = ROBOT_SIDES) -> tuple[str, ...]:
    """nutron-cli `policy_contract.patch_axis_names` re-derived (rbyte cannot import
    nutron-cli; the test asserts equality when a checkout is present): per-side 13
    names, side-major -- `left.joint1 .. left.pinky, right.joint1 .. right.pinky`.
    """
    return tuple(f"{side}.{name}" for side in sides for name in AXIS_NAMES)


def _recorder_order(sides: Sequence[str]) -> list[str]:
    """The recorder's names for the same 26-d order (`left.arm.0 .. right.hand.5`)."""
    return [
        f"{side}.{part}.{i}"
        for side in sides
        for part, n in (("arm", len(JOINT_NAMES)), ("hand", len(FINGER_NAMES)))
        for i in range(n)
    ]


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
class RobotArm:
    """One arm + hand's streams, every stream on the MCAP `publish_time` clock."""

    side: str
    measured_ts: npt.NDArray[np.int64]
    measured: npt.NDArray[np.float64]  # (N, 7)
    command_ts: npt.NDArray[np.int64]
    command: npt.NDArray[np.float64]  # (N, 7)
    hand_ts: npt.NDArray[np.int64]
    hand_counts: npt.NDArray[np.int64]  # (N, 6) raw 0..1000
    tactile: list[Any]
    #: recorder reset spans as OFFSETS from recording start (MCAP U outcome.json)
    hand_reset_spans_offset_ns: list[list[int]] | None = None

    @property
    def hand(self) -> npt.NDArray[np.float64]:
        """`robot[.{side}].hand.command / 1000` -- convert's `hand.val`."""
        return self.hand_counts.astype(np.float64) / HAND_SCALE


@dataclass
class RobotEpisode:
    """One robot recording: one arm (single-arm) or both (bimanual).

    `arms` is keyed by side in `ROBOT_SIDES` order: the active side alone for a
    single-arm take, `left` and `right` for a `bus_bimanual` one. The single-arm
    accessors (`measured_ts`, `hand`, ...) delegate to the one arm and refuse a
    bimanual episode.
    """

    name: str
    arms: dict[str, RobotArm]
    cam_ts: dict[str, npt.NDArray[np.int64]]
    cam_idx: dict[str, npt.NDArray[np.int64]]
    bimanual: bool = False
    recording_started_ns: int | None = None

    @property
    def sides(self) -> tuple[str, ...]:
        return tuple(self.arms)

    @property
    def active_side(self) -> str:
        return BIMANUAL_ACTIVE_HAND if self.bimanual else self._single().side

    def _single(self) -> RobotArm:
        if self.bimanual or len(self.arms) != 1:
            msg = f"{self.name}: bimanual episode, use ep.arms[side]"
            raise AttributeError(msg)
        (arm,) = self.arms.values()
        return arm

    @property
    def measured_ts(self) -> npt.NDArray[np.int64]:
        return self._single().measured_ts

    @property
    def measured(self) -> npt.NDArray[np.float64]:
        return self._single().measured

    @property
    def command_ts(self) -> npt.NDArray[np.int64]:
        return self._single().command_ts

    @property
    def command(self) -> npt.NDArray[np.float64]:
        return self._single().command

    @property
    def hand_ts(self) -> npt.NDArray[np.int64]:
        return self._single().hand_ts

    @property
    def hand_counts(self) -> npt.NDArray[np.int64]:
        return self._single().hand_counts

    @property
    def tactile(self) -> list[Any]:
        return self._single().tactile

    @property
    def hand_reset_spans_offset_ns(self) -> list[list[int]] | None:
        return self._single().hand_reset_spans_offset_ns

    @property
    def hand(self) -> npt.NDArray[np.float64]:
        """`robot.hand.command / 1000` -- convert's `ep.hand.val`."""
        return self._single().hand


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


def _reset_sources(
    metadata: Mapping[str, Mapping[str, str]],
    outcome: Mapping[str, Any],
    side: str | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """(MCAP copy, outcome.json copy) of the reset record (`convert._reset_sources`).

    Single-arm (`side=None`): MCAP `hand_reset` and outcome.json `hand_reset`.
    Bimanual: MCAP `hand_reset.{side}` and outcome.json `hand_reset.by_side.{side}`
    ONLY -- a bimanual take's top-level outcome `hand_reset` is a union summary.
    """
    if side is None:
        return metadata.get(_HAND_RESET_KEY, {}), outcome.get(_HAND_RESET_KEY) or {}
    by_side = (outcome.get(_HAND_RESET_KEY) or {}).get(_HAND_RESET_BY_SIDE) or {}
    return metadata.get(f"{_HAND_RESET_KEY}.{side}", {}), by_side.get(side) or {}


def _reset_spans_offset(
    metadata: Mapping[str, Mapping[str, str]],
    outcome: Mapping[str, Any],
    side: str | None = None,
) -> list[list[int]]:
    """`convert.hand_reset_spans`, but the UNION of both copies (brief: MCAP U disk)."""
    spans: list[list[int]] = []
    for source in _reset_sources(metadata, outcome, side):
        value = source.get("spans_ns") or "[]"
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = []
        if isinstance(value, list):
            spans += [[int(s), int(e)] for s, e in value]
    return spans


def _is_bimanual(path: Path, episode: Mapping[str, str]) -> bool:
    """`convert.episode_sides`: a `bus_bimanual` take must have `active_hand=both`.

    Raises:
        ValueError: on a `bus_bimanual` recording with one active hand (refused
            rather than half-filled, as in convert).
    """
    if episode.get("capture_mode") != BIMANUAL_CAPTURE_MODE:
        return False
    active = episode.get("active_hand")
    if active != BIMANUAL_ACTIVE_HAND:
        msg = (
            f"{path}: capture_mode {BIMANUAL_CAPTURE_MODE} with active_hand "
            f"{active!r}; only {BIMANUAL_ACTIVE_HAND!r} is supported"
        )
        raise ValueError(msg)
    return True


def _check_bimanual_metadata(path: Path, episode: Mapping[str, str]) -> None:
    """`convert.check_bimanual_metadata`: the recorder's 26-d order IS the layout.

    `state_order` / `action_order` use the recorder's names (`left.arm.0 ..
    right.hand.5`); index for index they must be `layout_names()`. The
    `state_*_source` keys are deliberately not read: the recorder declares tactile
    positions as the hand state, every consumer uses `hand_prev` (contract rule 2).

    Raises:
        ValueError: on any mismatch.
    """
    want = _recorder_order(ROBOT_SIDES)
    problems = []
    for key in ("state_order", "action_order"):
        got = [x for x in str(episode.get(key, "")).split(",") if x]
        if got != want:
            problems.append(f"episode {key} {got} != canonical {want}")
    problems.extend(
        f"{key} {episode[key]} != {len(want)}"
        for key in ("state_dimension", "action_dimension")
        if key in episode and int(episode[key]) != len(want)
    )
    fingers = [x for x in str(episode.get("hand_channel_order", "")).split(",") if x]
    if fingers != list(FINGER_NAMES):
        problems.append(f"hand_channel_order {fingers} != {list(FINGER_NAMES)}")
    if episode.get("arm_units") != _RECORDER_ARM_UNITS:
        problems.append(f"arm_units {episode.get('arm_units')!r}")
    if episode.get("hand_units") != _RECORDER_HAND_UNITS:
        problems.append(f"hand_units {episode.get('hand_units')!r}")
    if problems:
        msg = f"{path}: bimanual metadata is not the canonical layout: " + "; ".join(
            problems
        )
        raise ValueError(msg)


def _read_arm(
    path: Path,
    raw: Mapping[str, list[tuple[int, Any]]],
    side: str | None,
    *,
    arm_side: str,
    reset_spans: list[list[int]],
) -> RobotArm:
    """One arm's four streams (`side=None`: the single-arm `robot.*` topics).

    Raises:
        ValueError: on a missing stream or a malformed message; for a bimanual
            arm also on hand counts outside [0, 1000] or tactile rows tagged
            with the other side.
    """
    topics = side_topics(side)

    def joints(topic: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        ts, rows = _sorted_by_publish(list(raw[topic]))
        if not len(ts):
            msg = f"{path}: no {topic} messages"
            raise ValueError(msg)
        val = np.asarray([list(r.positions) for r in rows], dtype=np.float64)
        if val.shape[1:] != (7,):
            msg = f"{path}: {topic} positions are not 7-wide: {val.shape}"
            raise ValueError(msg)
        return ts, val

    measured_ts, measured = joints(topics["measured"])
    command_ts, command = joints(topics["command"])

    hand_ts, hand_rows = _sorted_by_publish(list(raw[topics["hand"]]))
    if not len(hand_ts):
        msg = f"{path}: no {topics['hand']} messages"
        raise ValueError(msg)
    hand_counts = np.asarray([list(r.counts) for r in hand_rows], dtype=np.int64)
    if hand_counts.shape[1:] != (6,):
        msg = f"{path}: {topics['hand']} counts are not 6-wide"
        raise ValueError(msg)

    _, tactile = _sorted_by_publish(list(raw[topics["tactile"]]))

    if side is not None:
        # contract rule 2: the hand part of state/action is counts/1000 in [0, 1]
        lo, hi = int(hand_counts.min()), int(hand_counts.max())
        if lo < 0 or hi > HAND_COUNT_MAX:
            msg = (
                f"{path}: {topics['hand']} counts outside "
                f"[0, {HAND_COUNT_MAX}]: {lo}..{hi}"
            )
            raise ValueError(msg)
        wrong = {
            str(getattr(r, "side", "") or "")
            for r in tactile
            if (getattr(r, "side", "") or "") not in {"", side}
        }
        if wrong:
            msg = (
                f"{path}: {topics['tactile']} carries rows tagged side {sorted(wrong)}"
            )
            raise ValueError(msg)

    return RobotArm(
        side=arm_side,
        measured_ts=measured_ts,
        measured=measured,
        command_ts=command_ts,
        command=command,
        hand_ts=hand_ts,
        hand_counts=hand_counts,
        tactile=tactile,
        hand_reset_spans_offset_ns=reset_spans,
    )


def read_episode(  # ruff:ignore[complex-structure, too-many-locals]
    path: PathLike[str] | str,
) -> RobotEpisode:
    """Read `data.mcap` (+ the sibling `outcome.json` if present).

    A `bus_bimanual` recording (MCAP `episode.capture_mode`) is read from the
    per-side topics; anything else from the single-arm `robot.*` topics.

    Raises:
        ValueError: on a missing stream or a malformed message.
    """
    path = Path(path)
    with path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        metadata = {r.name: dict(r.metadata) for r in reader.iter_metadata()}
    episode = metadata.get("episode", {})
    bimanual = _is_bimanual(path, episode)
    if bimanual:
        _check_bimanual_metadata(path, episode)
    arm_topics = (
        [t for side in ROBOT_SIDES for t in side_topics(side).values()]
        if bimanual
        else list(side_topics(None).values())
    )
    topics = arm_topics + [f"observation.images.{c}" for c in ROBOT_CAMERAS]
    raw: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    with path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=topics
        ):
            raw[channel.topic].append((message.publish_time, decoded))

    _camera_sources(metadata)

    outcome_path = path.parent / "outcome.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    if bimanual:
        arms = {
            side: _read_arm(
                path,
                raw,
                side,
                arm_side=side,
                reset_spans=_reset_spans_offset(metadata, outcome, side),
            )
            for side in ROBOT_SIDES
        }
    else:
        # the side is checked after the outcome (below); "left" is a placeholder
        arms = {
            "left": _read_arm(
                path,
                raw,
                None,
                arm_side="left",
                reset_spans=_reset_spans_offset(metadata, outcome),
            )
        }

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

    started = episode.get("recording_started_at_unix_ns")
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
    if not bimanual:
        active = str(episode.get("active_hand") or "left")
        if active not in ROBOT_SIDES:
            msg = f"{path}: active_hand {active!r} not in {ROBOT_SIDES}"
            raise ValueError(msg)
        arms["left"].side = active
        arms = {active: arms["left"]}

    return RobotEpisode(
        name=path.parent.name,
        arms=arms,
        cam_ts=cam_ts,
        cam_idx=cam_idx,
        bimanual=bimanual,
        recording_started_ns=int(started) if started else None,
    )


# ------------------------------------------------------------------ alignment


@dataclass
class SideAligned:
    """`convert.SideAligned`: one arm's own validity and ZOH indices."""

    valid: npt.NDArray[np.bool_]  # this side alone (fresh, in range, hand_prev fresh)
    j_measured: npt.NDArray[np.intp]
    j_command: npt.NDArray[np.intp]
    j_hand: npt.NDArray[np.intp]
    j_hand_prev: npt.NDArray[np.intp]


@dataclass
class Aligned:
    """`convert.Aligned`: per-base-frame alignment of one episode.

    `valid` is the AND over the episode's sides (contract rule 8) AND the
    side-camera match. `per_side[side]` holds each arm's own validity and
    indices (keyed by the episode's sides). The top-level `j_*` are that one
    arm's indices for a single-arm episode and None for a bimanual one.
    """

    valid: npt.NDArray[np.bool_]
    j_measured: npt.NDArray[np.intp] | None
    j_command: npt.NDArray[np.intp] | None
    j_hand: npt.NDArray[np.intp] | None
    j_hand_prev: npt.NDArray[np.intp] | None
    side_pick: dict[str, npt.NDArray[np.intp]]
    side_dt: dict[str, npt.NDArray[np.int64]]
    per_side: dict[str, SideAligned]


def align_side(arm: RobotArm, t: npt.NDArray[np.int64]) -> SideAligned:
    """`convert.align_side`, line for line: one arm's rule at base times `t`."""
    valid = np.ones(len(t), dtype=bool)
    js = {}
    for name, ts in (
        ("measured", arm.measured_ts),
        ("command", arm.command_ts),
        ("hand", arm.hand_ts),
    ):
        j, age = zoh(ts, t)
        valid &= (age <= MAX_STALE_NS) & (t <= ts[-1])
        js[name] = j

    j_hand_prev, hand_prev_age = zoh(arm.hand_ts, t - PREV_NS)
    valid &= hand_prev_age <= MAX_STALE_NS
    return SideAligned(valid, js["measured"], js["command"], js["hand"], j_hand_prev)


def align(ep: RobotEpisode) -> Aligned:
    """`convert.align`, line for line (one rule per side, ANDed)."""
    t = ep.cam_ts["base"]
    per_side = {side: align_side(arm, t) for side, arm in ep.arms.items()}
    valid = np.ones(len(t), dtype=bool)
    for sa in per_side.values():
        valid &= sa.valid

    side_pick, side_dt = {}, {}
    for camera in ("side_left", "side_right"):
        pick, dt = nearest(ep.cam_ts[camera], t)
        valid &= dt <= SIDE_TOL_NS
        side_pick[camera], side_dt[camera] = pick, dt

    single = None if ep.bimanual else next(iter(per_side.values()))
    return Aligned(
        valid=valid,
        j_measured=None if single is None else single.j_measured,
        j_command=None if single is None else single.j_command,
        j_hand=None if single is None else single.j_hand,
        j_hand_prev=None if single is None else single.j_hand_prev,
        side_pick=side_pick,
        side_dt=side_dt,
        per_side=per_side,
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


def _reset_spans(
    ep: RobotEpisode, arm: RobotArm, motor: hf.MotorTimeline
) -> npt.NDArray[np.int64]:
    """Recorder spans (MCAP U outcome.json) UNION self-detected, absolute ns."""
    spans = [hf.reset_spans_from_timeline(motor)]
    if arm.hand_reset_spans_offset_ns and ep.recording_started_ns is not None:
        spans.append(
            hf.reset_spans_from_record(
                arm.hand_reset_spans_offset_ns, ep.recording_started_ns
            )
        )
    elif arm.hand_reset_spans_offset_ns:
        logger.warning(
            "hand_reset spans recorded without recording_started_at_unix_ns; "
            "using self-detection only",
            episode=ep.name,
            side=arm.side,
        )
    return np.concatenate(spans, axis=0).astype(np.int64).reshape(-1, 2)


def _hand_columns(
    prefix: str, tokens: hf.TokenBlocks, keep_idx: npt.NDArray[np.intp]
) -> dict[str, npt.NDArray[Any]]:
    return {
        f"{prefix}current": tokens.blocks["current"][keep_idx],
        f"{prefix}pos_err": tokens.blocks["pos_err"][keep_idx],
        f"{prefix}pos": tokens.blocks["pos"][keep_idx],
        f"{prefix}tip": tokens.blocks["tip"][keep_idx],
        f"{prefix}age": tokens.blocks["hand_age"][keep_idx, 0],
        f"{prefix}motor_ok": tokens.motor_ok[keep_idx],
        f"{prefix}tip_ok": tokens.tip_ok[keep_idx],
    }


def build_rows(  # ruff:ignore[too-many-locals, too-many-statements, complex-structure]
    ep: RobotEpisode,
    *,
    chunk_size: int = 100,
    max_pad_steps: int | None = None,
    min_run: int = MIN_RUN,
    camera_cond: npt.NDArray[np.float32] | None = None,
) -> Rows:
    """All row columns for one episode (see the module docstring).

    Raises:
        ValueError: if `chunk_size` < 1, or (bimanual) a hand value of the
            state or chunk falls outside [0, 1].
    """
    if chunk_size < 1:
        msg = "chunk_size must be >= 1"
        raise ValueError(msg)
    max_pad = chunk_size // 2 if max_pad_steps is None else max_pad_steps
    t = ep.cam_ts["base"]
    n = len(t)
    a = align(ep)
    sides = ep.sides

    tokens: dict[str, hf.TokenBlocks] = {}
    motor_rows: dict[str, int] = {}
    poison = np.zeros(n, dtype=bool)
    for side, arm in ep.arms.items():
        motor = hf.motor_timeline(arm.tactile)
        tips = hf.tip_timeline(arm.tactile)
        commands = hf.make_command_timeline(arm.hand_ts, arm.hand_counts)
        tokens[side] = hf.build_tokens(t, motor, commands, tips)
        motor_rows[side] = len(motor)
        # a reset on EITHER hand poisons the (shared) row
        poison |= hf.reset_poison(t, _reset_spans(ep, arm, motor))

    grid = grid_index(t)
    duplicate = duplicate_grid(grid)
    usable = a.valid & ~poison & ~duplicate
    runs = runs_of(usable, min_run)

    offsets = chunk_offsets_ns(chunk_size)
    hands = {side: arm.hand for side, arm in ep.arms.items()}

    keep: list[int] = []
    chunks: list[npt.NDArray[np.float64]] = []
    pads: list[npt.NDArray[np.bool_]] = []
    dropped_pad = 0
    for start, stop in runs:
        rows = np.arange(start, stop)
        t_end = t[stop - 1]
        tk = t[rows, None] + offsets[None, :]  # (R, C)
        stale = np.zeros(tk.shape, dtype=bool)
        per_side: list[npt.NDArray[np.float64]] = []
        for side, arm in ep.arms.items():
            jc, age_c = zoh(arm.command_ts, tk.reshape(-1))
            jh, age_h = zoh(arm.hand_ts, tk.reshape(-1))
            stale |= ((age_c > MAX_STALE_NS) | (age_h > MAX_STALE_NS)).reshape(tk.shape)
            per_side.append(
                np.concatenate(
                    [
                        arm.command[jc].reshape(*tk.shape, 7),
                        hands[side][jh].reshape(*tk.shape, 6),
                    ],
                    axis=-1,
                )
            )
        chunk = np.stack(per_side, axis=2)  # (R, C, S_ep, 13)
        # a pad is a SUFFIX: the first padded step (past the run end, or EITHER
        # side stale) and everything after it -- one union mask for all sides
        pad = np.logical_or.accumulate((tk > t_end) | stale, axis=1)
        pad[:, 0] = False  # k = 0 is t itself, a valid frame by construction
        # every side holds its last real step (of the UNION) through the tail
        last = np.maximum((~pad).sum(axis=1) - 1, 0)
        held = chunk[np.arange(len(rows)), last]  # (R, S_ep, 13)
        chunk = np.where(pad[..., None, None], held[:, None], chunk)
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
        else np.zeros((0, chunk_size, len(sides), ACTION_DIM))
    )
    pad_all = (
        np.concatenate(pads, axis=0) if pads else np.zeros((0, chunk_size), dtype=bool)
    )

    state = np.zeros((r, len(ROBOT_SIDES), ACTION_DIM), dtype=np.float32)
    action = np.zeros((r, chunk_size, len(ROBOT_SIDES), ACTION_DIM), dtype=np.float32)
    side_valid = np.zeros((r, len(ROBOT_SIDES)), dtype=bool)
    for k, (side, arm) in enumerate(ep.arms.items()):
        s = ROBOT_SIDES.index(side)
        sa = a.per_side[side]
        state[:, s] = np.concatenate(
            [
                arm.measured[sa.j_measured[keep_idx]],
                hands[side][sa.j_hand_prev[keep_idx]],
            ],
            axis=-1,
        ).astype(np.float32)
        action[:, :, s] = chunk_all[:, :, k].astype(np.float32)
        side_valid[:, s] = True

    if ep.bimanual:
        hand_dims = slice(len(JOINT_NAMES), ACTION_DIM)
        for name, values in (("state", state), ("action.chunk", action)):
            h = values[..., hand_dims]
            if h.size and (h.min() < 0 or h.max() > 1):
                msg = f"{ep.name}: {name} hand dims outside [0, 1]"
                raise ValueError(msg)

    cond = (
        np.zeros((len(ROBOT_CAMERAS), 13), dtype=np.float32)
        if camera_cond is None
        else np.asarray(camera_cond, dtype=np.float32).reshape(len(ROBOT_CAMERAS), 13)
    )

    hand_columns: dict[str, npt.NDArray[Any]] = {}
    if ep.bimanual:
        for side in sides:
            hand_columns |= _hand_columns(f"hand.{side}.", tokens[side], keep_idx)
    else:
        hand_columns = _hand_columns("hand.", tokens[sides[0]], keep_idx)

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
        **hand_columns,
        "camera_cond": np.broadcast_to(cond, (r, *cond.shape)).copy(),
        "camera_cond.placeholder": np.full(r, camera_cond is None),
    }

    def per_hand(values: Mapping[str, int]) -> int | dict[str, int]:
        return dict(values) if ep.bimanual else values[sides[0]]

    stats = {
        "frames": n,
        "valid": int(a.valid.sum()),
        "reset_poison": int(poison.sum()),
        "duplicate_grid": int(duplicate.sum()),
        "runs": runs,
        "rows": r,
        "dropped_pad": dropped_pad,
        "padded_rows": int(pad_all.any(axis=1).sum()),
        "hand_ok_rows": per_hand({
            side: int(tokens[side].motor_ok[keep_idx].sum()) for side in sides
        }),
        "tactile_rows": per_hand({
            side: len(arm.tactile) for side, arm in ep.arms.items()
        }),
        "motor_rows": per_hand(motor_rows),
    }
    if ep.bimanual:
        stats["side_valid"] = {
            side: int(a.per_side[side].valid.sum()) for side in sides
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
