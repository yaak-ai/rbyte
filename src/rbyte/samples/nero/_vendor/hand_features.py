"""Revo2 hand feedback -> ACT ``observation.environment_state``, as pure numpy.

THE ONE BUILDER. training/convert.py (offline, every base frame of an episode at
once) and jetson/policy_source.py (online, one frame per inference) both import
this file and call the SAME function, ``build_blocks``; the online entry point
``build_env`` is literally ``build_blocks`` on a length-1 time array, row 0. There
is no second implementation to drift, which is the whole parity argument: a
dataset row and a served observation can only differ if their INPUTS differ.

Importable on the Orin's py3.10 with numpy alone -- no torch, no lerobot, no
protobuf. Rows are read by attribute OR by key, so the same parser takes a
``recording_pb2.HandTactile`` from an MCAP and a dict off the action bus.

WHAT THE VECTOR IS. Per selected group, a 1 s history on a fixed 10 Hz grid
going back from the frame time t: slot k sits at ``t - k * 100 ms`` (k = 0..9,
newest first) and holds the newest VALID sample whose host time is AT OR BEFORE
the slot -- ``convert.zoh`` semantics exactly (searchsorted side="right" - 1), so
a sample stamped 1 ns after a slot is NOT in it. Groups, canonical order:

    current   6 motor currents, clipped to +-1000 then / 1000          (60)
    pos_err   robot command (ZOH at the SAMPLE's host time) - measured
              position, / 1000                                         (60)
    pos       6 measured positions, / 1000                             (60)
    tip       5 fingertips x (normal, tangential) force, clipped / 1000 (100)
              -- ablation only, OFF unless asked for
    age       newest motor sample's age at t, seconds                   (1)
    valid     1.0 for a row the builder accepted                        (1)

``age`` and ``valid`` are not selectable: they come along with any hand group.
Within a group the flattening is slot-major, newest slot first, then channel
(``current.t-000ms.thumb_flex``, ..., ``current.t-900ms.pinky``); ``env_columns``
is the authority and the contract carries it.

REFUSAL, NOT PADDING. A row is refused (reason string, zero vector) unless the
NEWEST slot (k = 0, at t) holds a valid sample no older than MAX_HAND_STALE_NS
(250 ms), every OLDER slot (k = 1..9) holds one no older than MAX_SLOT_STALE_NS
(500 ms) at THAT slot's time -- both inclusive, like convert's 100 ms state ZOH --
none of those samples is a power-cycle reset sample, and the robot command exists
at every slot's sample time. Two limits because the two ends mean different
things: slot 0 is what the hand is doing NOW and must be tight; an older slot is
context, and the rig's measured arrival is bursty (inter-sample gaps p50 71 / p90
208 / p99 375 / max 483 ms), so a 250 ms limit there refused ~24-41 % of frames
over gaps the policy would see at serving time anyway. A repeated older slot is
in-distribution; a 300 ms-old "now" is not. Training drops refused frames (when a
hand group is enabled); serving holds. Refusal codes keep the two apart:
hand-stale (slot 0) vs hand-history-gap (an older slot).

TIMESTAMPS ARE INT64 NANOSECONDS, both sides, and floats are rejected. At 1.79e9 s
a float64 second is ~240 ns coarse, so a float frame clock cannot even express
"1 ns after the slot". Serving takes the relay frame's ``st_mtime_ns``.

    from jetson import hand_features as hf            # training/ (py3.12)
    import hand_features as hf                        # jetson/   (py3.10)

    motor = hf.motor_timeline(tactile_rows)           # filter + stable sort
    tips = hf.tip_timeline(tactile_rows)
    commands = hf.make_command_timeline(cmd_t_ns, cmd_counts)   # RAW counts
    blocks = hf.build_blocks(frame_t_ns, motor, commands, tips)      # offline
    vector, refusal = hf.build_env(t_ns, motor, commands, groups, tips)  # online

THE PATCH FAMILY'S HAND TOKEN (contract v3, family "patch") is a second surface in
the same file -- newest sample only, + age + validity, refusal = no_hand, its own
HAND_TOKEN_SPEC_VERSION -- see the "newest-sample hand token" section:

    tokens = hf.build_tokens(frame_t_ns, motor, commands, tips)       # offline (rbyte)
    x = tokens.compose(groups)                                        # (N, token_dim)
    vec, valid, reason = hf.build_token(t_ns, motor, commands, groups, tips)  # online

VENDORED. rbyte cannot import nutron-cli, so it carries a VERBATIM copy of this
file pinned by SHA256 (docs/notes/HAND-FEATURES-VENDORING.md, hand_features_sync.py).
ANY edit here -- a comment included -- changes the hash: re-vendor into rbyte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------- constants ---

# Bumped on ANY change to a constant, the column layout, the clip/scale rules or
# the refusal rule: a checkpoint trained on version N must not be served with N+1.
#   1  every slot <= 250 ms old at its slot
#   2  slot 0 <= 250 ms, older slots <= 500 ms; reset pad + motor window widened
HAND_FEATURES_VERSION = 2

FINGER_NAMES = ("thumb_flex", "thumb_aux", "index", "middle", "ring", "pinky")
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
N_MOTORS = len(FINGER_NAMES)
N_TIPS = len(TIP_NAMES)

SLOT_NS = 100_000_000  # 10 Hz history grid
HISTORY_SLOTS = 10  # 1 s: slots at t, t-100 ms, ..., t-900 ms
HISTORY_NS = SLOT_NS * (HISTORY_SLOTS - 1)  # offset of the OLDEST slot
# Max age of the sample filling a slot, measured AT THAT SLOT. Inclusive (<=), the
# convert.MAX_STALE_NS convention. Two limits, one per end of the history:
#
# MAX_HAND_STALE_NS, slot 0 (the frame time t): 2.5 sample periods at the
# measured ~10 Hz collector rate absorb one dropped read; the ~3 Hz polling
# corpora fail it, which is deliberate (a "now" that is 300 ms late is not the
# signal the policy learns). Also the aux target's freshness bound (current_now).
#
# MAX_SLOT_STALE_NS, slots 1..9: the rig's tactile arrival is BURSTY, not a 10 Hz
# clock -- measured inter-sample gaps p50 71 / p90 208 / p99 375 / max 483 ms -- so
# holding the older slots to 250 ms refused a quarter to two fifths of otherwise
# good frames over gaps serving sees just the same. 500 ms clears the measured max
# with margin and still refuses a real outage (a dead collector, a re-home's
# 1-1.5 s silence). A long gap shows up as repeated slots, which is exactly what
# the policy is trained on and will be served.
MAX_HAND_STALE_NS = 250_000_000
MAX_SLOT_STALE_NS = 500_000_000

# Counts -> model units. Positions, commands and currents share the Revo2's
# 0..1000 count scale; the firmware clamps a healthy grasp current at exactly
# +-1000 and only a power-cycle re-home exceeds it (1930-2229 measured), so the
# clip removes reset outliers BEFORE any statistic sees them.
SCALE = 1000.0
CURRENT_CLIP = 1000
# Fingertip force units are raw sensor counts (uint16, ~0..100 observed at
# light contact). The clip is a guess at "far beyond any real contact", there so
# a sensor glitch cannot reach the normalizer; revisit once there is contact data.
TIP_SCALE = 1000.0
TIP_CLIP = 5000

# record.py's HandResetDetector signature, mirrored (not imported: record.py is the
# kit recorder and pulls protobuf/mcap): all six positions exactly 0 and
# max |current| above this. A healthy open hand reads all-zero positions all the
# time; only a re-homing hand also draws more than a clamped grasp ever does.
RESET_CURRENT = 1500
RESET_COALESCE_NS = 2_000_000_000
# Everything an accept/refuse decision on the motor stream can depend on lies in
# [t - MOTOR_WINDOW_NS, t]: the oldest slot (t - 900 ms) may be filled by a sample
# up to MAX_SLOT_STALE_NS older still, so 1.4 s. The one exception is pos_err's
# command lookup: a command is a SETPOINT that holds until replaced, so the newest
# command at or before the oldest slot's sample may be arbitrarily old (the arm's
# deadband legitimately sends nothing while the hand is steady). A serving ring
# must keep at least one command at or before ``t - MOTOR_WINDOW_NS``; see
# ``trim_commands``.
MOTOR_WINDOW_NS = HISTORY_NS + MAX_SLOT_STALE_NS

# The per-slot limit, newest first: what ``fresh`` compares each slot's age to.
SLOT_STALE_NS = np.array(
    [MAX_HAND_STALE_NS] + [MAX_SLOT_STALE_NS] * (HISTORY_SLOTS - 1), dtype=np.int64
)
# The parity argument needs every accepted slot's sample INSIDE the window: a
# slot-k sample of age a sits at t - k * SLOT_NS - a. If a limit ever reached past
# the window, an offline row (whole episode in memory) could accept what a
# serving ring has already evicted -- and _age_detail's "none" would lie.
assert all(
    int(SLOT_STALE_NS[k]) <= MOTOR_WINDOW_NS - k * SLOT_NS for k in range(HISTORY_SLOTS)
), "a slot's staleness limit reaches past MOTOR_WINDOW_NS"

# Padding around a reset span for training-frame poisoning. Before: the 1-1.5 s
# telemetry gap that precedes the detected samples (motor_valid false, nothing
# for the detector to see). After: the re-home tail, plus one full motor window
# (history + the older-slot staleness allowance) so no accepted frame's history
# -- whose oldest slot may hold a sample from t - 1.4 s -- reaches back into the
# span.
RESET_PAD_BEFORE_NS = 2_000_000_000
RESET_PAD_AFTER_NS = 2_000_000_000 + MOTOR_WINDOW_NS

SELECTABLE_GROUPS = ("current", "pos_err", "pos", "tip")
AUTO_GROUPS = ("age", "valid")
GROUP_ORDER = SELECTABLE_GROUPS + AUTO_GROUPS
DEFAULT_GROUPS = ("current", "pos_err", "pos")  # decision 2; tip is opt-in
MOTOR_GROUPS = ("current", "pos_err", "pos")

# Per-dim std floor applied when the env normalization stats are composed, so a
# near-constant channel (pinky current, a repeated slot) cannot turn sensor noise
# into a large normalized value, and the ALWAYS-1 validity flag (std 0 in
# training) normalizes to 0 instead of dividing by 1e-8. Model units.
STD_FLOOR = {
    "current": 0.02,
    "pos_err": 0.01,
    "pos": 0.01,
    "tip": 0.01,
    "age": 0.02,
    "valid": 1.0,
}

# The future-current auxiliary target (decision 7): 6 dims appended after the 13
# action dims, one per motor, value = ``current_now`` at the chunk step's frame.
AUX_NAMES = tuple(f"aux_current.{name}" for name in FINGER_NAMES)

# convert.py's dataset columns. Deliberately NOT ``observation.*``: lerobot types
# every observation.* key as a STATE input, and only the exact key
# observation.environment_state is ENV. The training wrapper composes the selected
# groups into that key at load time, so a group can be switched without reconvert.
DATASET_PREFIX = "hand_env."
DATASET_MOTOR_OK = DATASET_PREFIX + "motor_ok"
DATASET_TIP_OK = DATASET_PREFIX + "tip_ok"
DATASET_CURRENT_NOW = DATASET_PREFIX + "current_now"
DATASET_CURRENT_NOW_OK = DATASET_PREFIX + "current_now_ok"
DATASET_RESET_POISON = DATASET_PREFIX + "reset_poison"

# Refusal reason codes. Each refusal is "<code>" or "<code>:<detail>"; serving
# prefixes "held:policy-" (e.g. held:policy-hand-stale:312ms). "hand-stale:none"
# means no valid sample inside the window at all (never seen, or long gone).
# hand-stale = slot 0 older than MAX_HAND_STALE_NS (the hand's "now" is late);
# hand-history-gap:t-<k>ms:<age> = slot 0 fine, the FIRST older slot whose sample
# is older than MAX_SLOT_STALE_NS at that slot (an outage inside the last second).
REFUSE_STALE = "hand-stale"
REFUSE_HISTORY = "hand-history-gap"
REFUSE_RESET = "hand-reset"
REFUSE_NO_COMMAND = "hand-no-command"
REFUSE_TIP_STALE = "tip-stale"
REFUSE_TIP_HISTORY = "tip-history-gap"

_NO_SAMPLE_AGE = np.iinfo(np.int64).max


# --------------------------------------------------------------- the groups ---


def normalize_groups(groups: str | Sequence[str] | None) -> tuple[str, ...]:
    """User selection -> the selected groups in CANONICAL order.

    Accepts "current,pos_err", a list, "none"/""/None. Order and duplicates in the
    input do not matter -- the env layout is fixed by GROUP_ORDER, never by how a
    flag happened to be spelled. Unknown and auto groups are refused.
    """
    if groups is None:
        return ()
    if isinstance(groups, str):
        items = [g.strip() for g in groups.split(",")]
    else:
        items = [str(g).strip() for g in groups]
    items = [g for g in items if g and g != "none"]
    bad = [g for g in items if g not in SELECTABLE_GROUPS]
    if bad:
        raise ValueError(
            f"unknown hand env group(s) {bad}; selectable: {', '.join(SELECTABLE_GROUPS)}"
            + (" (age/valid come with any group)" if set(bad) & set(AUTO_GROUPS) else "")
        )
    return tuple(g for g in SELECTABLE_GROUPS if g in items)


def env_groups(selected: Sequence[str]) -> tuple[str, ...]:
    """Selected groups plus the automatic age/valid columns; () for no selection."""
    sel = normalize_groups(selected)
    return sel + AUTO_GROUPS if sel else ()


def _slot_tag(k: int) -> str:
    return f"t-{k * SLOT_NS // 1_000_000:03d}ms"


def group_columns(group: str) -> list[str]:
    """Column names of one group, in its flattened order."""
    slots = [_slot_tag(k) for k in range(HISTORY_SLOTS)]
    if group in MOTOR_GROUPS:
        return [f"{group}.{s}.{f}" for s in slots for f in FINGER_NAMES]
    if group == "tip":
        return [
            f"tip.{s}.{kind}.{tip}"
            for s in slots
            for kind in ("normal", "tangential")
            for tip in TIP_NAMES
        ]
    if group == "age":
        return ["age_s"]
    if group == "valid":
        return ["valid"]
    raise ValueError(f"unknown hand env group {group!r}")


def group_width(group: str) -> int:
    return len(group_columns(group))


def env_columns(selected: Sequence[str]) -> list[str]:
    """Every column of observation.environment_state for this selection, in order."""
    return [c for g in env_groups(selected) for c in group_columns(g)]


def env_dim(selected: Sequence[str]) -> int:
    return len(env_columns(selected))


def group_slices(selected: Sequence[str]) -> dict[str, slice]:
    """Where each group sits inside the composed env vector (for metrics/ablation)."""
    out: dict[str, slice] = {}
    at = 0
    for g in env_groups(selected):
        w = group_width(g)
        out[g] = slice(at, at + w)
        at += w
    return out


def std_floor_vector(selected: Sequence[str]) -> np.ndarray:
    """Per-dim std floor for the composed env vector, float32."""
    parts = [np.full(group_width(g), STD_FLOOR[g], dtype=np.float32) for g in env_groups(selected)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def apply_std_floor(std: Any, selected: Sequence[str]) -> np.ndarray:
    """``max(std, floor)`` per dim, float32. Training writes the result into the
    checkpoint's normalizer stats, so serving never applies a floor of its own."""
    arr = np.asarray(std, dtype=np.float32)
    floor = std_floor_vector(selected)
    if arr.shape != floor.shape:
        raise ValueError(f"env std has shape {arr.shape}, selection needs {floor.shape}")
    return np.maximum(arr, floor)


def constants() -> dict[str, Any]:
    """Every number the layout and the refusal rule depend on, for the contract."""
    return {
        "hand_features_version": HAND_FEATURES_VERSION,
        "finger_names": list(FINGER_NAMES),
        "tip_names": list(TIP_NAMES),
        "slot_ns": SLOT_NS,
        "history_slots": HISTORY_SLOTS,
        "max_hand_stale_ns": MAX_HAND_STALE_NS,
        "max_slot_stale_ns": MAX_SLOT_STALE_NS,
        "scale": SCALE,
        "current_clip": CURRENT_CLIP,
        "tip_scale": TIP_SCALE,
        "tip_clip": TIP_CLIP,
        "reset_current": RESET_CURRENT,
        "std_floor": dict(STD_FLOOR),
    }


def env_spec(selected: Sequence[str]) -> dict[str, Any]:
    """The JSON-able description of one env selection.

    convert.py writes ``env_spec(SELECTABLE_GROUPS)['constants']`` into the dataset
    meta, training copies ``env_spec(groups)`` into the checkpoint config, export
    copies it into ``<name>_contract.json``, and every server compares the copy it
    was handed with a fresh ``env_spec`` from ITS OWN import of this file. Equal
    dicts or a refusal; there is no partial match.
    """
    sel = normalize_groups(selected)
    return {
        "groups": list(sel),
        "auto_groups": list(AUTO_GROUPS) if sel else [],
        "columns": env_columns(sel),
        "dim": env_dim(sel),
        "constants": constants(),
    }


# --------------------------------------------------------------- timelines ---


@dataclass(frozen=True)
class MotorTimeline:
    """Valid motor samples only, sorted by host time. Raw int64 counts."""

    t_ns: np.ndarray  # (N,) int64, non-decreasing
    positions: np.ndarray  # (N, 6) int64
    currents: np.ndarray  # (N, 6) int64

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])


@dataclass(frozen=True)
class TipTimeline:
    """Complete fingertip reads only, sorted by touch host time. Raw int64."""

    t_ns: np.ndarray  # (M,) int64
    normal: np.ndarray  # (M, 5) int64
    tangential: np.ndarray  # (M, 5) int64

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])


@dataclass(frozen=True)
class CommandTimeline:
    """The ROBOT-SIDE hand command (robot.hand.command), raw 0..1000 counts.

    Raw counts, not convert's /1000 floats: pos_err is computed in integer counts
    and scaled once, and ``x / 1000 * 1000`` is not exact in floating point.
    """

    t_ns: np.ndarray  # (K,) int64
    counts: np.ndarray  # (K, 6) int64

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])


def _as_ns(t: Any, what: str) -> np.ndarray:
    arr = np.asarray(t)
    if arr.size == 0:  # np.asarray([]) is float64; an empty ring is not a float clock
        return np.zeros(arr.shape, dtype=np.int64)
    if arr.dtype.kind == "f":
        raise TypeError(
            f"{what}: float timestamps refused -- host times are int64 ns "
            "(a float64 second at 1.8e9 s cannot resolve 1 ns)"
        )
    if arr.dtype.kind not in "iu":
        raise TypeError(f"{what}: timestamps must be integer ns, got {arr.dtype}")
    return arr.astype(np.int64)


def _stable_sorted(t_ns: np.ndarray, *cols: np.ndarray) -> tuple[np.ndarray, ...]:
    """Stable sort by time: equal stamps keep their input (= arrival) order, so the
    ZOH (side="right" - 1) picks the LAST row of a tie on both sides."""
    order = np.argsort(t_ns, kind="stable")
    return (t_ns[order],) + tuple(c[order] for c in cols)


def _int_matrix(values: Any, width: int, what: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.zeros((0, width), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{what}: expected (N, {width}), got {arr.shape}")
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)) or not np.array_equal(arr, np.round(arr)):
            raise ValueError(f"{what}: raw counts must be integers")
    return arr.astype(np.int64)


def make_motor_timeline(t_ns: Any, positions: Any, currents: Any) -> MotorTimeline:
    """From arrays that are ALREADY valid (6 positions + 6 currents per row)."""
    t = _as_ns(t_ns, "motor t_ns").reshape(-1)
    pos = _int_matrix(positions, N_MOTORS, "motor positions")
    cur = _int_matrix(currents, N_MOTORS, "motor currents")
    if not (len(t) == len(pos) == len(cur)):
        raise ValueError(f"motor rows disagree: {len(t)} t, {len(pos)} pos, {len(cur)} cur")
    t, pos, cur = _stable_sorted(t, pos, cur)
    return MotorTimeline(t, pos, cur)


def make_tip_timeline(t_ns: Any, normal: Any, tangential: Any) -> TipTimeline:
    t = _as_ns(t_ns, "tip t_ns").reshape(-1)
    nrm = _int_matrix(normal, N_TIPS, "tip normal")
    tan = _int_matrix(tangential, N_TIPS, "tip tangential")
    if not (len(t) == len(nrm) == len(tan)):
        raise ValueError("tip rows disagree in length")
    t, nrm, tan = _stable_sorted(t, nrm, tan)
    return TipTimeline(t, nrm, tan)


def make_command_timeline(t_ns: Any, counts: Any) -> CommandTimeline:
    t = _as_ns(t_ns, "command t_ns").reshape(-1)
    cnt = _int_matrix(counts, N_MOTORS, "command counts")
    if len(t) != len(cnt):
        raise ValueError("command rows disagree in length")
    t, cnt = _stable_sorted(t, cnt)
    return CommandTimeline(t, cnt)


def _get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def motor_timeline(rows: Iterable[Any]) -> MotorTimeline:
    """Raw ``robot.hand.tactile`` rows (proto or bus dict) -> the motor timeline.

    THE FILTER BOTH SIDES RUN. A row counts only if motor_valid is true, it has a
    host ``sample_time_ns`` > 0, and it carries exactly six positions AND six
    currents. ~23 % of rows on the first collector-era take are motor_valid false
    with EMPTY arrays (a touch-only or failed read) yet still carry a sample time;
    letting one of those be "the newest sample" would make the age a lie and the
    indexing either crash or zero-fill. Reset samples are KEPT here (they are
    valid reads) and refused by the builder, so both sides refuse them identically.
    """
    ts: list[int] = []
    pos: list[list[int]] = []
    cur: list[list[int]] = []
    for row in rows:
        if not _get(row, "motor_valid", False):
            continue
        t = int(_get(row, "sample_time_ns", 0) or 0)
        p = list(_get(row, "positions", ()) or ())
        c = list(_get(row, "currents", ()) or ())
        if t <= 0 or len(p) != N_MOTORS or len(c) != N_MOTORS:
            continue
        ts.append(t)
        pos.append([int(v) for v in p])
        cur.append([int(v) for v in c])
    return make_motor_timeline(np.asarray(ts, dtype=np.int64), pos, cur)


def tip_timeline(rows: Iterable[Any]) -> TipTimeline:
    """Raw tactile rows -> complete fingertip reads on their OWN clock.

    The fingertip read has its own host stamp (``touch_sample_time_ns``, 0 = no new
    touch items in this row) and its own presence bits; it never borrows the motor
    timeline's. A row counts only if the stamp is > 0 and all five fingertips are
    present with five normal and five tangential values.
    """
    ts: list[int] = []
    nrm: list[list[int]] = []
    tan: list[list[int]] = []
    for row in rows:
        t = int(_get(row, "touch_sample_time_ns", 0) or 0)
        present = list(_get(row, "touch_finger_present", ()) or ())
        n = list(_get(row, "normal_force", ()) or ())
        g = list(_get(row, "tangential_force", ()) or ())
        if t <= 0 or len(present) != N_TIPS or not all(present):
            continue
        if len(n) != N_TIPS or len(g) != N_TIPS:
            continue
        ts.append(t)
        nrm.append([int(v) for v in n])
        tan.append([int(v) for v in g])
    return make_tip_timeline(np.asarray(ts, dtype=np.int64), nrm, tan)


def trim_commands(commands: CommandTimeline, t_ns: int) -> CommandTimeline:
    """The smallest suffix of ``commands`` that still answers every pos_err lookup
    for frames at or after ``t_ns``: everything newer than the window start plus
    the one newest command at or before it (the setpoint in force then)."""
    start = int(t_ns) - MOTOR_WINDOW_NS
    j = int(np.searchsorted(commands.t_ns, start, side="right")) - 1
    j = max(j, 0)
    return CommandTimeline(commands.t_ns[j:], commands.counts[j:])


# ---------------------------------------------------------------- builder ---


def _zoh(src_ts: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``convert.zoh`` on any-shaped ``t``: (index clipped >= 0, has-sample, age ns).

    side="right" - 1: a sample stamped exactly at t IS the hold; one stamped 1 ns
    later is not. Age is +inf (int64 max) where nothing precedes t.
    """
    if src_ts.shape[0] == 0:
        zero = np.zeros(t.shape, dtype=np.int64)
        return zero, np.zeros(t.shape, dtype=bool), np.full(t.shape, _NO_SAMPLE_AGE, np.int64)
    j = np.searchsorted(src_ts, t, side="right") - 1
    has = j >= 0
    jc = np.maximum(j, 0)
    age = np.where(has, t - src_ts[jc], _NO_SAMPLE_AGE)
    return jc, has, age


def is_reset_sample(positions: Any, currents: Any) -> np.ndarray:
    """record.py's power-cycle signature per row: all six positions == 0 AND
    max |current| > RESET_CURRENT. Raw counts in, bool out (shape [..])."""
    pos = np.asarray(positions)
    cur = np.asarray(currents)
    if pos.size == 0:
        return np.zeros(pos.shape[:-1], dtype=bool)
    return np.all(pos == 0, axis=-1) & (np.max(np.abs(cur), axis=-1) > RESET_CURRENT)


def _scaled(values: np.ndarray, lo: float, hi: float, scale: float) -> np.ndarray:
    # One fixed order on both sides: int64 clip, float64 divide, ONE float32 cast.
    return (np.clip(values, lo, hi).astype(np.float64) / scale).astype(np.float32)


def _age_detail(has: Any, age_ns: Any, k: int) -> str:
    """Refusal detail for a slot: the sample's age at the slot, or "none".

    "none" ALSO covers a sample older than the motor window (for k = 9 that is
    exactly the older-slot limit; see the SLOT_STALE_NS assert). Without that, the
    same frame would read "hand-stale:5012ms" offline (the whole episode is in
    memory) and "hand-stale:none" on a serving ring that already evicted it --
    the refusal TEXT, not just the verdict, must be a function of the window.
    """
    if not bool(has) or int(age_ns) > MOTOR_WINDOW_NS - k * SLOT_NS:
        return "none"
    return f"{int(age_ns) / 1e6:.0f}ms"


@dataclass(frozen=True)
class EnvBlocks:
    """Every group's block for N frame times, plus why each refused row refused.

    ``blocks[g]`` is (N, group_width(g)) float32 for EVERY group in GROUP_ORDER,
    zero on refused rows. Motor groups and tip have separate verdicts because
    they are separately selectable: a tip-stale row is fine for a model without tip.
    """

    t_ns: np.ndarray  # (N,) int64
    blocks: dict[str, np.ndarray]
    motor_ok: np.ndarray  # (N,) bool
    tip_ok: np.ndarray  # (N,) bool; all False when no tip timeline was given
    motor_reason: list[str | None]
    tip_reason: list[str | None]

    def ok(self, selected: Sequence[str]) -> np.ndarray:
        """Rows usable for this selection. No selection -> every row."""
        sel = normalize_groups(selected)
        out = np.ones(self.t_ns.shape[0], dtype=bool)
        if sel:
            out &= self.motor_ok
        if "tip" in sel:
            out &= self.tip_ok
        return out

    def refusal(self, selected: Sequence[str], i: int) -> str | None:
        sel = normalize_groups(selected)
        if sel and self.motor_reason[i] is not None:
            return self.motor_reason[i]
        if "tip" in sel and self.tip_reason[i] is not None:
            return self.tip_reason[i]
        return None


def _motor_verdicts(
    t: np.ndarray, motor: MotorTimeline, commands: CommandTimeline
) -> tuple[np.ndarray, list[str | None], np.ndarray, np.ndarray]:
    """-> (ok, reasons, slot sample index (N, S), pos_err counts (N, S, 6))."""
    n = t.shape[0]
    slots = t[:, None] - SLOT_NS * np.arange(HISTORY_SLOTS, dtype=np.int64)[None, :]
    j, has, age = _zoh(motor.t_ns, slots)
    # Per-slot limit: 250 ms at slot 0, 500 ms at every older slot (SLOT_STALE_NS).
    fresh = has & (age <= SLOT_STALE_NS[None, :])

    if len(motor):
        reset_rows = is_reset_sample(motor.positions, motor.currents)
        reset = np.any(reset_rows[j] & has, axis=1)
        sample_t = motor.t_ns[j]
        jc, has_cmd, _ = _zoh(commands.t_ns, sample_t)
        # A command is a setpoint, held until replaced: no age bound (see
        # MOTOR_WINDOW_NS). It only has to EXIST at the sample's host time.
        cmd_ok = np.all(has_cmd | ~has, axis=1)
        if len(commands):
            pos_err = commands.counts[jc] - motor.positions[j]
        else:
            pos_err = np.zeros(slots.shape + (N_MOTORS,), dtype=np.int64)
    else:
        reset = np.zeros(n, dtype=bool)
        cmd_ok = np.ones(n, dtype=bool)
        pos_err = np.zeros(slots.shape + (N_MOTORS,), dtype=np.int64)

    ok = np.all(fresh, axis=1) & ~reset & cmd_ok
    reasons: list[str | None] = [None] * n
    for i in np.flatnonzero(~ok):
        if not fresh[i, 0]:
            reasons[i] = f"{REFUSE_STALE}:{_age_detail(has[i, 0], age[i, 0], 0)}"
        elif not fresh[i].all():
            k = int(np.flatnonzero(~fresh[i])[0])
            reasons[i] = f"{REFUSE_HISTORY}:{_slot_tag(k)}:{_age_detail(has[i, k], age[i, k], k)}"
        elif reset[i]:
            reasons[i] = REFUSE_RESET
        else:
            reasons[i] = REFUSE_NO_COMMAND
    return ok, reasons, j, pos_err


def _tip_verdicts(
    t: np.ndarray, tips: TipTimeline | None
) -> tuple[np.ndarray, list[str | None], np.ndarray]:
    n = t.shape[0]
    if tips is None:
        return np.zeros(n, dtype=bool), [f"{REFUSE_TIP_STALE}:none"] * n, np.zeros(
            (n, HISTORY_SLOTS), dtype=np.int64
        )
    slots = t[:, None] - SLOT_NS * np.arange(HISTORY_SLOTS, dtype=np.int64)[None, :]
    j, has, age = _zoh(tips.t_ns, slots)
    # The motor rule on the fingertip clock: same two limits, same refusal split.
    fresh = has & (age <= SLOT_STALE_NS[None, :])
    ok = np.all(fresh, axis=1)
    reasons: list[str | None] = [None] * n
    for i in np.flatnonzero(~ok):
        if not fresh[i, 0]:
            reasons[i] = f"{REFUSE_TIP_STALE}:{_age_detail(has[i, 0], age[i, 0], 0)}"
        else:
            k = int(np.flatnonzero(~fresh[i])[0])
            reasons[i] = f"{REFUSE_TIP_HISTORY}:{_slot_tag(k)}:{_age_detail(has[i, k], age[i, k], k)}"
    return ok, reasons, j


def build_blocks(
    t_ns: Any,
    motor: MotorTimeline,
    commands: CommandTimeline,
    tips: TipTimeline | None = None,
) -> EnvBlocks:
    """All env groups at each frame time in ``t_ns`` (int64 ns), vectorised.

    The ONLY builder: ``build_env`` (serving) calls this with one time. Every value
    is a pure function of the timelines restricted to [t - MOTOR_WINDOW_NS, t]
    (plus the setpoint command in force at the oldest sample), so a serving ring
    holding that much reproduces an offline row bit for bit.
    """
    t = _as_ns(t_ns, "frame t_ns").reshape(-1)
    n = t.shape[0]
    motor_ok, motor_reason, j, pos_err = _motor_verdicts(t, motor, commands)
    tip_ok, tip_reason, jt = _tip_verdicts(t, tips)

    blocks: dict[str, np.ndarray] = {}
    if len(motor):
        cur = motor.currents[j]  # (N, S, 6)
        pos = motor.positions[j]
        blocks["current"] = _scaled(cur, -CURRENT_CLIP, CURRENT_CLIP, SCALE).reshape(n, -1)
        blocks["pos"] = (pos.astype(np.float64) / SCALE).astype(np.float32).reshape(n, -1)
        blocks["pos_err"] = (pos_err.astype(np.float64) / SCALE).astype(np.float32).reshape(n, -1)
        newest_age = t - motor.t_ns[j[:, 0]]
        blocks["age"] = (newest_age.astype(np.float64) / 1e9).astype(np.float32).reshape(n, 1)
    else:
        for g in MOTOR_GROUPS + ("age",):
            blocks[g] = np.zeros((n, group_width(g)), dtype=np.float32)
    blocks["valid"] = motor_ok.astype(np.float32).reshape(n, 1)
    if tips is not None and len(tips):
        nrm = _scaled(tips.normal[jt], 0, TIP_CLIP, TIP_SCALE)  # (N, S, 5)
        tan = _scaled(tips.tangential[jt], 0, TIP_CLIP, TIP_SCALE)
        blocks["tip"] = np.concatenate((nrm, tan), axis=2).reshape(n, -1)
    else:
        blocks["tip"] = np.zeros((n, group_width("tip")), dtype=np.float32)

    for g in MOTOR_GROUPS + ("age",):
        blocks[g][~motor_ok] = 0.0
    blocks["tip"][~tip_ok] = 0.0
    return EnvBlocks(
        t_ns=t,
        blocks={g: np.ascontiguousarray(blocks[g]) for g in GROUP_ORDER},
        motor_ok=motor_ok,
        tip_ok=tip_ok,
        motor_reason=motor_reason,
        tip_reason=tip_reason,
    )


def compose(blocks: Mapping[str, Any], selected: Sequence[str]) -> np.ndarray:
    """Concatenate the selected groups (+ age/valid) in canonical order, float32.

    ``blocks`` maps group -> (N, width) or (width,) arrays: an ``EnvBlocks.blocks``
    or the per-group dataset columns. Shape-checked against the layout.
    """
    groups = env_groups(selected)
    if not groups:
        first = next(iter(blocks.values()), None)
        lead = () if first is None else np.asarray(first).shape[:-1]
        return np.zeros(lead + (0,), dtype=np.float32)
    parts = []
    for g in groups:
        arr = np.asarray(blocks[g], dtype=np.float32)
        if arr.shape[-1] != group_width(g):
            raise ValueError(f"group {g!r} has width {arr.shape[-1]}, layout says {group_width(g)}")
        parts.append(arr)
    return np.concatenate(parts, axis=-1)


def build_env(
    t_ns: int,
    motor: MotorTimeline,
    commands: CommandTimeline,
    selected: Sequence[str],
    tips: TipTimeline | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """ONE frame, the serving call: -> (env vector float32 (dim,), None) or
    (None, refusal). ``build_blocks`` on a length-1 array; nothing else."""
    if isinstance(t_ns, (float, np.floating)):
        raise TypeError("frame t_ns must be int ns (use st_mtime_ns), not float seconds")
    sel = normalize_groups(selected)
    if not sel:
        return np.zeros(0, dtype=np.float32), None
    b = build_blocks(np.asarray([t_ns], dtype=np.int64), motor, commands, tips)
    reason = b.refusal(sel, 0)
    if reason is not None:
        return None, reason
    return compose({g: v[0] for g, v in b.blocks.items()}, sel), None


# ------------------------------------------- newest-sample hand token (patch) ---
#
# THE PATCH FAMILY'S HAND INPUT. The causal patch policy (rmind NeroPatchPolicy,
# contract v3 family "patch") sees ONE hand token per 10 Hz frame and keeps the
# history itself, in its KV cache -- so it reads only what ACT's slot 0 holds:
# the newest valid sample at or before t, plus that sample's age and a validity
# flag. Same timelines, same filters, same ZOH (side="right" - 1), same clip and
# /1000 scaling as ``build_blocks``; on an accepted row every value group is
# BIT-IDENTICAL to ACT's ``<group>.t-000ms.*`` columns (test_hand_features pins
# it). rbyte (offline, vendored copy of THIS file) calls ``build_tokens`` on every
# frame of an episode; policy_source (online) calls ``build_token``, which is
# ``build_tokens`` on a length-1 array, row 0. One builder, as above.
#
# A separate surface with its own version so ACT is untouched: HAND_FEATURES_VERSION,
# ``constants()``, ``env_spec()`` and every ACT column are unchanged, and a contract
# v2 artifact still diffs equal. HAND_TOKEN_SPEC_VERSION is bumped on ANY change to
# a token constant, the token column layout, the age scaling or the token refusal
# rule; ``token_spec`` carries both versions, so an HAND_FEATURES_VERSION bump (a
# shared scale or filter changed) also invalidates a token artifact.
#
# REFUSAL MEANS no_hand, NOT DROP OR HOLD. A KV stream cannot skip a 10 Hz frame,
# so a refused row is still a row: the vector is all zeros with valid = 0 and the
# model substitutes its learned no_hand token (training and serving alike). The
# reason string is for counters (serving logs ``hand:no_hand:<reason>``). Judged on
# the newest sample ONLY -- an older gap is irrelevant to a model without slots:
#
#   hand-stale[:<age>|:none]  newest valid motor sample older than MAX_HAND_STALE_NS
#                             (250 ms, inclusive) at t, or none in the window
#   hand-reset                a power-cycle reset sample (is_reset_sample) anywhere
#                             in [t - TOKEN_RESET_LOOKBACK_NS, t] -- the newest
#                             sample itself, or the re-home tail after one (the
#                             2 s "after" pad of the reset-span rule, without
#                             ACT's history allowance, which a token has no use for)
#   hand-no-command           no robot hand command at or before the newest
#                             sample's host time (pos_err undefined). Checked for
#                             EVERY selection, like ACT, so the motor verdict never
#                             depends on which groups are selected
#   tip-stale[:<age>|:none]   tip selected and the newest complete fingertip read
#                             older than 250 ms (its own clock, as in ACT)
#
# Rows inside a recorder reset span (``reset_poison``, offline metadata) are a
# TRAINING-side exclusion exactly as for ACT; the builder only sees the motor data.
#
# Columns, canonical order (selectable groups in SELECTABLE_GROUPS order, then the
# two automatic ones):
#
#   current.now.<finger>             6   clip +-1000, / 1000
#   pos_err.now.<finger>             6   (command ZOH at the sample time - pos) / 1000
#   pos.now.<finger>                 6   / 1000
#   tip.now.<normal|tangential>.<tip> 10  clip 0..5000, / 1000   (ablation only)
#   hand_age                         1   newest motor sample age / 250 ms, clip [0, 2]
#   hand_valid                       1   1.0 accepted, 0.0 refused
#
# On a refused row EVERY column is 0 (age included), so the vector cannot leak a
# stale value past the validity flag.

HAND_TOKEN_SPEC_VERSION = 1
TOKEN_DEFAULT_GROUPS = ("current", "pos_err")  # pos optional, tip ablation only
TOKEN_AUTO_COLUMNS = ("hand_age", "hand_valid")
TOKEN_AGE_SCALE_NS = 250_000_000  # age column = age_ns / this, clipped
TOKEN_AGE_CLIP = 2.0
# Re-home tail after a reset sample during which the token is refused: the "after"
# pad of the reset-span rule minus ACT's motor-window allowance (= 2 s).
TOKEN_RESET_LOOKBACK_NS = RESET_PAD_AFTER_NS - MOTOR_WINDOW_NS
# Everything a token verdict/value can depend on lies in [t - TOKEN_WINDOW_NS, t]
# (plus, for pos_err, the setpoint command in force at the newest sample -- keep one
# command at or before ``t - TOKEN_WINDOW_NS``, see ``trim_commands_token``). A
# serving ring must hold at least this much motor history.
TOKEN_WINDOW_NS = max(MAX_HAND_STALE_NS, TOKEN_RESET_LOOKBACK_NS)
assert TOKEN_RESET_LOOKBACK_NS == 2_000_000_000


def token_group_columns(group: str) -> list[str]:
    """Column names of one token group (selectable groups only), flattened order."""
    if group in MOTOR_GROUPS:
        return [f"{group}.now.{f}" for f in FINGER_NAMES]
    if group == "tip":
        return [f"tip.now.{kind}.{tip}" for kind in ("normal", "tangential") for tip in TIP_NAMES]
    raise ValueError(f"unknown hand token group {group!r}; selectable: {', '.join(SELECTABLE_GROUPS)}")


def token_columns(selected: Sequence[str]) -> list[str]:
    """THE token layout authority: every column of the hand token for this
    selection, in order. No selection -> [] (the model has no hand token)."""
    sel = normalize_groups(selected)
    if not sel:
        return []
    return [c for g in sel for c in token_group_columns(g)] + list(TOKEN_AUTO_COLUMNS)


def token_dim(selected: Sequence[str]) -> int:
    return len(token_columns(selected))


def token_slices(selected: Sequence[str]) -> dict[str, slice]:
    """Where each group (and hand_age / hand_valid) sits inside the token vector."""
    out: dict[str, slice] = {}
    at = 0
    sel = normalize_groups(selected)
    if not sel:
        return out
    for g in sel:
        w = len(token_group_columns(g))
        out[g] = slice(at, at + w)
        at += w
    for name in TOKEN_AUTO_COLUMNS:
        out[name] = slice(at, at + 1)
        at += 1
    return out


def token_constants() -> dict[str, Any]:
    """Every number the token layout, scaling and refusal rule depend on."""
    return {
        "hand_token_spec_version": HAND_TOKEN_SPEC_VERSION,
        "hand_features_version": HAND_FEATURES_VERSION,
        "finger_names": list(FINGER_NAMES),
        "tip_names": list(TIP_NAMES),
        "max_hand_stale_ns": MAX_HAND_STALE_NS,
        "reset_lookback_ns": TOKEN_RESET_LOOKBACK_NS,
        "reset_current": RESET_CURRENT,
        "scale": SCALE,
        "current_clip": CURRENT_CLIP,
        "tip_scale": TIP_SCALE,
        "tip_clip": TIP_CLIP,
        "age_scale_ns": TOKEN_AGE_SCALE_NS,
        "age_clip": TOKEN_AGE_CLIP,
    }


def token_spec(selected: Sequence[str]) -> dict[str, Any]:
    """JSON-able description of one hand-token selection; the contract v3 ``hand``
    block. Serving compares the copy in the contract with a fresh ``token_spec``
    from its own import: equal dicts or a refusal. rbyte writes the same dict into
    its dataset meta, rmind into the checkpoint config."""
    sel = normalize_groups(selected)
    if not sel:
        raise ValueError("token_spec needs at least one hand group (no hand token -> hand: null)")
    return {
        "kind": "newest_sample",
        "groups": list(sel),
        "columns": token_columns(sel),
        "dim": token_dim(sel),
        "constants": token_constants(),
    }


@dataclass(frozen=True)
class TokenBlocks:
    """Every token group for N frame times, plus validity and refusal reasons.

    ``blocks[g]`` is (N, width) float32 for EVERY selectable group plus
    ``hand_age`` / ``hand_valid`` (N, 1); motor-refused rows are zero in all of
    them. ``motor_ok`` and ``tip_ok`` are separate because tip is separately
    selectable, so ``blocks["hand_valid"]`` is the MOTOR verdict only: the token
    for a selection comes from ``compose`` (which also applies tip's verdict when
    tip is selected), never from concatenating blocks by hand.
    """

    t_ns: np.ndarray  # (N,) int64
    blocks: dict[str, np.ndarray]
    motor_ok: np.ndarray  # (N,) bool
    tip_ok: np.ndarray  # (N,) bool; all False when no tip timeline was given
    motor_reason: list[str | None]
    tip_reason: list[str | None]
    age_ns: np.ndarray  # (N,) int64 newest motor sample age at t; int64 max = none

    def ok(self, selected: Sequence[str]) -> np.ndarray:
        """hand_valid for this selection (no selection -> all False: no token)."""
        sel = normalize_groups(selected)
        if not sel:
            return np.zeros(self.t_ns.shape[0], dtype=bool)
        out = self.motor_ok.copy()
        if "tip" in sel:
            out &= self.tip_ok
        return out

    def refusal(self, selected: Sequence[str], i: int) -> str | None:
        sel = normalize_groups(selected)
        if not sel:
            return None
        if self.motor_reason[i] is not None:
            return self.motor_reason[i]
        if "tip" in sel and self.tip_reason[i] is not None:
            return self.tip_reason[i]
        return None

    def compose(self, selected: Sequence[str]) -> np.ndarray:
        """(N, token_dim) float32 for this selection; a row refused for THIS
        selection (e.g. tip-stale with tip selected) is all zero, valid = 0."""
        sel = normalize_groups(selected)
        n = self.t_ns.shape[0]
        if not sel:
            return np.zeros((n, 0), dtype=np.float32)
        ok = self.ok(sel)
        parts = [self.blocks[g] for g in sel] + [self.blocks["hand_age"], ok.astype(np.float32)[:, None]]
        out = np.concatenate(parts, axis=1).astype(np.float32, copy=True)
        out[~ok] = 0.0
        return out


def build_tokens(
    t_ns: Any,
    motor: MotorTimeline,
    commands: CommandTimeline,
    tips: TipTimeline | None = None,
) -> TokenBlocks:
    """Newest-sample hand token at each frame time in ``t_ns`` (int64 ns), vectorised.

    THE ONLY token builder: ``build_token`` (serving) calls this with one time.
    Every value is a pure function of the timelines restricted to
    [t - TOKEN_WINDOW_NS, t] plus the setpoint command in force at the newest
    sample, so a serving ring holding that much reproduces an offline row bit for bit.
    """
    t = _as_ns(t_ns, "frame t_ns").reshape(-1)
    n = t.shape[0]
    j, has, age = _zoh(motor.t_ns, t)
    fresh = has & (age <= MAX_HAND_STALE_NS)

    if len(motor):
        reset_rows = is_reset_sample(motor.positions, motor.currents)
        cum = np.concatenate(([0], np.cumsum(reset_rows, dtype=np.int64)))
        lo = np.searchsorted(motor.t_ns, t - TOKEN_RESET_LOOKBACK_NS, side="left")
        hi = np.searchsorted(motor.t_ns, t, side="right")
        reset = (cum[hi] - cum[lo]) > 0
        jc, has_cmd, _ = _zoh(commands.t_ns, motor.t_ns[j])
        cmd_ok = has_cmd | ~has
        if len(commands):
            pos_err = commands.counts[jc] - motor.positions[j]  # (N, 6) int64
        else:
            pos_err = np.zeros((n, N_MOTORS), dtype=np.int64)
        cur = _scaled(motor.currents[j], -CURRENT_CLIP, CURRENT_CLIP, SCALE)
        pos = (motor.positions[j].astype(np.float64) / SCALE).astype(np.float32)
        perr = (pos_err.astype(np.float64) / SCALE).astype(np.float32)
    else:
        reset = np.zeros(n, dtype=bool)
        cmd_ok = np.ones(n, dtype=bool)
        cur = np.zeros((n, N_MOTORS), dtype=np.float32)
        pos = np.zeros((n, N_MOTORS), dtype=np.float32)
        perr = np.zeros((n, N_MOTORS), dtype=np.float32)

    motor_ok = fresh & ~reset & cmd_ok
    motor_reason: list[str | None] = [None] * n
    for i in np.flatnonzero(~motor_ok):
        if not fresh[i]:
            motor_reason[i] = f"{REFUSE_STALE}:{_token_age_detail(has[i], age[i])}"
        elif reset[i]:
            motor_reason[i] = REFUSE_RESET
        else:
            motor_reason[i] = REFUSE_NO_COMMAND

    age_col = np.clip(
        np.where(has, age, 0).astype(np.float64) / TOKEN_AGE_SCALE_NS, 0.0, TOKEN_AGE_CLIP
    ).astype(np.float32)

    if tips is not None and len(tips):
        jt, has_t, age_t = _zoh(tips.t_ns, t)
        tip_ok = has_t & (age_t <= MAX_HAND_STALE_NS)
        nrm = _scaled(tips.normal[jt], 0, TIP_CLIP, TIP_SCALE)
        tan = _scaled(tips.tangential[jt], 0, TIP_CLIP, TIP_SCALE)
        tip = np.concatenate((nrm, tan), axis=1)
        tip_reason: list[str | None] = [
            None if tip_ok[i] else f"{REFUSE_TIP_STALE}:{_token_age_detail(has_t[i], age_t[i])}"
            for i in range(n)
        ]
    else:
        tip_ok = np.zeros(n, dtype=bool)
        tip = np.zeros((n, 2 * N_TIPS), dtype=np.float32)
        tip_reason = [f"{REFUSE_TIP_STALE}:none"] * n

    for arr in (cur, pos, perr, age_col):
        arr[~motor_ok] = 0.0
    tip[~tip_ok] = 0.0
    blocks = {
        "current": cur,
        "pos_err": perr,
        "pos": pos,
        "tip": tip,
        "hand_age": age_col.reshape(n, 1),
        "hand_valid": motor_ok.astype(np.float32).reshape(n, 1),
    }
    return TokenBlocks(
        t_ns=t,
        blocks={k: np.ascontiguousarray(v) for k, v in blocks.items()},
        motor_ok=motor_ok,
        tip_ok=tip_ok,
        motor_reason=motor_reason,
        tip_reason=tip_reason,
        age_ns=np.where(has, age, _NO_SAMPLE_AGE).astype(np.int64),
    )


def _token_age_detail(has: Any, age_ns: Any) -> str:
    """Refusal detail for the newest sample: its age, or "none" when there is no
    sample inside TOKEN_WINDOW_NS (so offline and a serving ring say the same)."""
    if not bool(has) or int(age_ns) > TOKEN_WINDOW_NS:
        return "none"
    return f"{int(age_ns) / 1e6:.0f}ms"


def build_token(
    t_ns: int,
    motor: MotorTimeline,
    commands: CommandTimeline,
    selected: Sequence[str],
    tips: TipTimeline | None = None,
) -> tuple[np.ndarray, bool, str | None]:
    """ONE frame, the serving call: -> (token float32 (token_dim,), valid, reason).

    ALWAYS a vector: a refused row is all zeros with hand_valid 0 and ``reason``
    set, and the caller feeds it as is (the model swaps in no_hand). ``build_tokens``
    on a length-1 array; nothing else."""
    if isinstance(t_ns, (float, np.floating)):
        raise TypeError("frame t_ns must be int ns (use st_mtime_ns), not float seconds")
    sel = normalize_groups(selected)
    if not sel:
        return np.zeros(0, dtype=np.float32), False, None
    b = build_tokens(np.asarray([t_ns], dtype=np.int64), motor, commands, tips)
    vec = b.compose(sel)[0]
    reason = b.refusal(sel, 0)
    return vec, reason is None, reason


def trim_commands_token(commands: CommandTimeline, t_ns: int) -> CommandTimeline:
    """``trim_commands`` for the token's window: everything newer than
    ``t - TOKEN_WINDOW_NS`` plus the setpoint in force at that instant."""
    start = int(t_ns) - TOKEN_WINDOW_NS
    j = int(np.searchsorted(commands.t_ns, start, side="right")) - 1
    j = max(j, 0)
    return CommandTimeline(commands.t_ns[j:], commands.counts[j:])


# -------------------------------------------------------------- aux target ---


def current_now(t_ns: Any, motor: MotorTimeline) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame current as the robot would REPORT it by time t: (N, 6) float32
    (clipped, /1000) and an ok mask (N,).

    The aux target for chunk step k of a frame at t is this column read at the
    frame t+k (the dataset wrapper loads it with the action's delta indices), i.e.
    the newest valid sample at or before that frame's time -- causal at every
    step, never a sample from after the frame it is attributed to. ok = such a
    sample exists, is no older than MAX_HAND_STALE_NS, and is not a reset sample;
    not-ok values are 0 and must be masked out of the loss per dim. The 250 ms
    bound, not the older slots' 500 ms: a target is a "now" reading, the slot-0
    analogue, and a 450 ms-old current is not what the hand drew at t+k.
    """
    t = _as_ns(t_ns, "frame t_ns").reshape(-1)
    j, has, age = _zoh(motor.t_ns, t)
    ok = has & (age <= MAX_HAND_STALE_NS)
    if not len(motor):
        return np.zeros((t.shape[0], N_MOTORS), dtype=np.float32), ok
    ok &= ~is_reset_sample(motor.positions[j], motor.currents[j])
    vals = _scaled(motor.currents[j], -CURRENT_CLIP, CURRENT_CLIP, SCALE)
    vals[~ok] = 0.0
    return vals, ok


# ----------------------------------------------------------- reset spans ---


def reset_spans_from_record(spans_offset_ns: Any, recording_started_ns: int | str) -> np.ndarray:
    """record.py's ``spans_ns`` -> absolute host-ns spans, (S, 2) int64.

    The recorder stores OFFSETS from ``episode.recording_started_at_unix_ns``
    (MCAP 'episode' metadata), not timestamps. A one-sample event has start == end.
    Both arguments may be the raw MCAP metadata strings ('[[s, e], ...]', '179...').
    """
    if isinstance(spans_offset_ns, (str, bytes)):  # MCAP metadata stores JSON text
        import json  # noqa: PLC0415

        spans_offset_ns = json.loads(spans_offset_ns or "[]")
    arr = np.asarray(spans_offset_ns if spans_offset_ns is not None else [], dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    arr = arr.reshape(-1, 2)
    return arr + np.int64(int(recording_started_ns))


def reset_spans_from_timeline(motor: MotorTimeline, coalesce_ns: int = RESET_COALESCE_NS) -> np.ndarray:
    """Self-detected reset spans (absolute host ns), coalesced like the recorder.

    The recorder's own record can be missing (an outcome.json rewrite, or a take
    older than the detector), so training ORs this in rather than trusting it.
    """
    if not len(motor):
        return np.zeros((0, 2), dtype=np.int64)
    ts = motor.t_ns[is_reset_sample(motor.positions, motor.currents)]
    spans: list[list[int]] = []
    for t in ts.tolist():
        if spans and t - spans[-1][1] <= coalesce_ns:
            spans[-1][1] = max(spans[-1][1], t)
        else:
            spans.append([t, t])
    return np.asarray(spans, dtype=np.int64).reshape(-1, 2)


def reset_poison(t_ns: Any, spans_abs_ns: Any) -> np.ndarray:
    """Frames inside any padded reset span ``[s - RESET_PAD_BEFORE_NS,
    e + RESET_PAD_AFTER_NS]`` (inclusive): (N,) bool.

    Covers the frame's own env history. A frame whose ACTION chunk runs into a span
    is a training-side concern (pad those steps with action_is_pad); this mask is
    the per-frame input for it.
    """
    t = _as_ns(t_ns, "frame t_ns").reshape(-1)
    spans = np.asarray(spans_abs_ns, dtype=np.int64).reshape(-1, 2)
    out = np.zeros(t.shape[0], dtype=bool)
    for s, e in spans.tolist():
        out |= (t >= s - RESET_PAD_BEFORE_NS) & (t <= e + RESET_PAD_AFTER_NS)
    return out
