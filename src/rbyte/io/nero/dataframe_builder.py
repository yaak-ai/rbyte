"""nero-arms MCAP -> 30 Hz camera-grid dataframe (data contract §3, §5, §6, §9)."""

from collections import defaultdict
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Final, NamedTuple, final

import numpy as np
import numpy.typing as npt
import polars as pl
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from pydantic import NonNegativeFloat, PositiveInt, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars

from rbyte.io.nero.calibration import NeroArmsCalibration
from rbyte.io.nero.disparity import (
    DISPARITY_METADATA_PREFIX,
    DISPARITY_TOPIC_PREFIX,
    DisparityDeclaration,
)
from rbyte.io.nero.rotation import canonicalize_quat, quat_slerp, quat_to_matrix
from rbyte.io.nero.schema import (
    CAMERAS,
    FINGERS,
    IMU_DIM,
    IMU_SEGMENTS,
    SIDES,
    STATE_DIM_QUAT,
    STATUS_SENSORS,
)

logger = get_logger(__name__)

__all__ = ["NeroArmsDataFrameBuilder"]

_TIMING_TOPIC: Final = "timing.rgmp"
_IMAGE_TOPIC_PREFIX: Final = "observation.images."
_HUB_ORIENTATION_TOPIC: Final = "hub.LTP_NED.orientation"
_NS_PER_MS: Final = 1e6
#: §3.4: one glove period at 83.4 Hz.
_GLOVE_PERIOD_MS: Final = 12.0
_POSE_WIDTH: Final = 7
_QUAT_WIDTH: Final = 4

type _Floats = dict[str, list[list[float]]]
type _Ints = dict[str, list[list[int]]]


class _Stream(NamedTuple):
    """A device-clock time base plus the host-clock arrival stamps for it.

    Device epochs are unrelated across devices (§3.2), so every stream is mapped
    onto a common (host) clock via a single constant offset estimated from
    `host_arrival_time_ns - device_timestamp_ns`.
    """

    device_ns: npt.NDArray[np.int64]
    host_ns: npt.NDArray[np.int64]
    #: percentile of the observed latency used as the offset estimate; 0 is the
    #: pure minimum-latency estimator. The mean is biased by transport jitter.
    percentile: float = 0.0

    @property
    def latency_ns(self) -> npt.NDArray[np.int64]:
        return self.host_ns - self.device_ns

    @property
    def offset_ns(self) -> int:
        return int(np.percentile(self.latency_ns, self.percentile, method="lower"))

    @property
    def common_ns(self) -> npt.NDArray[np.int64]:
        return self.device_ns + self.offset_ns

    @property
    def latency_spread_ms(self) -> float:
        latency = self.latency_ns

        return float(np.max(latency) - np.min(latency)) / _NS_PER_MS

    @property
    def offset_drift_ms(self) -> float:
        """Offset estimated on the first vs the second half of the episode.

        §3.2 asserts a per-episode constant offset is sufficient (drift < 7 ms);
        this is the number that claim should be checked against.
        """
        half = len(self.device_ns) // 2
        if half == 0:
            return 0.0

        first = _Stream(self.device_ns[:half], self.host_ns[:half], self.percentile)
        second = _Stream(self.device_ns[half:], self.host_ns[half:], self.percentile)

        return abs(first.offset_ns - second.offset_ns) / _NS_PER_MS


class _Camera(NamedTuple):
    """Any `ImageFrameIndex` stream -- an mp4 camera or the disparity mkv."""

    frame_index: npt.NDArray[np.int32]
    stream: _Stream


class _Resampling(NamedTuple):
    """The §3.3 resampling plan: reference grid + bracketing glove samples."""

    #: reference-camera timestamps on the common clock, extrapolation excluded
    grid_ns: npt.NDArray[np.int64]
    #: the reference camera's own `frame_index` for those timestamps
    frame_index: npt.NDArray[np.int32]
    lo: npt.NDArray[np.intp]
    hi: npt.NDArray[np.intp]
    #: interpolation parameter within `[lo, hi]`
    u: npt.NDArray[np.float64]
    #: whichever of `lo`/`hi` is closer in time -- for non-interpolable values
    nearest: npt.NDArray[np.intp]
    #: §3.4: distance to the nearest bracketing glove sample
    residual_ms: npt.NDArray[np.float64]

    def lerp(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u = self.u[:, None]

        return values[self.lo] * (1.0 - u) + values[self.hi] * u

    def slerp(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return canonicalize_quat(
            quat_slerp(
                canonicalize_quat(values[self.lo]),
                canonicalize_quat(values[self.hi]),
                self.u,
            )
        )


def _topic(side: str, name: str) -> str:
    return f"{side}.{name}"


@final
class NeroArmsDataFrameBuilder:
    """Build the per-episode 30 Hz dataframe for one nero-arms recording.

    Reads a `data.mcap`, aligns the glove stream onto the reference camera's
    device-clock grid (§3), canonicalises quaternions (§5.1), assembles the
    bimanual state/action blocks with a `side_valid` mask (§6) and the goal
    blocks (§9).

    Emitted columns (one row per reference-camera frame):

    | column                        | dtype                        |
    |-------------------------------|------------------------------|
    | `frame_index.{camera}`        | `Int32`                      |
    | `frame_index.disparity.{cam}` | `Int32`                      |
    | `state.pose`                  | `Array(Float32, (2, 46))`    |
    | `state.pose_rel_start`        | `Array(Float32, (2, 46))`    |
    | `action.future_state`         | `Array(Float32, (H, 2, 46))` |
    | `side_valid`                  | `Array(Boolean, 2)`          |
    | `align_residual_ms`           | `Float32`                    |
    | `camera_cond`                 | `Array(Float32, (3, 13))`    |
    | `camera_cond.placeholder`     | `Boolean`                    |
    | `goal.xyz`                    | `Array(Float32, (2, 3))`     |
    | `goal.frame_index.{camera}`   | `Int32`                      |
    | `state.imu`                   | `Array(Float32, (2, 42))`    |
    | `state.status_flags`          | `Array(Int32, (2, 8))`       |

    `state.pose` is the **storage** form of §5.2 (7 floats per pose, 46 per
    side). Use `rbyte.io.nero.state_quat_to_9d` to obtain the 60-dim
    model-facing form of §6.1 at the rmind input boundary.

    The last `action_horizon` rows of every episode are dropped: they have no
    full future chunk, and clamping or wrapping the chunk would fabricate data.

    `disparity_cameras` (§21, empty by default) adds the depth stream's own
    index columns and asserts the episode's declared stereo mode -- `subpixel`
    false above all. The pixels themselves are decoded by
    `rbyte.io.NeroArmsDisparityFrameSource`.
    """

    __name__ = __qualname__

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        cameras: Sequence[str] = CAMERAS,
        reference_camera: str = CAMERAS[0],
        sides: Sequence[str] = SIDES,
        action_horizon: PositiveInt = 6,
        align_tolerance_ms: NonNegativeFloat = _GLOVE_PERIOD_MS,
        camera_gap_tolerance_ms: NonNegativeFloat = 50.0,
        offset_percentile: NonNegativeFloat = 0.0,
        include_imu: bool = False,
        include_status_flags: bool = False,
        disparity_cameras: Sequence[str] = (),
    ) -> None:
        if reference_camera not in cameras:
            logger.error(msg := "`reference_camera` not in `cameras`")

            raise ValueError(msg)

        self._cameras = tuple(cameras)
        self._reference_camera = reference_camera
        self._sides = tuple(sides)
        self._disparity_cameras = tuple(disparity_cameras)
        self._action_horizon = action_horizon
        self._align_tolerance_ms = align_tolerance_ms
        self._camera_gap_tolerance_ms = camera_gap_tolerance_ms
        self._offset_percentile = offset_percentile
        self._include_imu = include_imu
        self._include_status_flags = include_status_flags

    def __call__(
        self, path: PathLike[str] | str, calibration_path: PathLike[str] | str
    ) -> pl.DataFrame:
        with bound_contextvars(path=str(path)):
            result = self._build(Path(path), Path(calibration_path))
            logger.debug("built dataframe", length=len(result))

            return result

    # ------------------------------------------------------------------ topics

    @property
    def _pose_topics(self) -> tuple[str, ...]:
        return (
            "arm_sensor.coil_pro.transform",
            *(f"{finger}_finger_sensor.hub.transform" for finger in FINGERS),
        )

    def _side_topics(self, side: str) -> tuple[str, ...]:
        return (
            *(_topic(side, topic) for topic in self._pose_topics),
            _topic(side, _HUB_ORIENTATION_TOPIC),
        )

    @property
    def _index_topics(self) -> dict[str, str]:
        """Column key -> `ImageFrameIndex` topic, reference camera first.

        §3/§21.4: every one of these is joined on **its own** `frame_index` --
        the disparity stream may have a different frame count from the mp4s just
        as the mp4s differ from each other.
        """
        return {
            camera: f"{_IMAGE_TOPIC_PREFIX}{camera}" for camera in self._cameras
        } | {
            f"disparity.{camera}": f"{DISPARITY_TOPIC_PREFIX}{camera}"
            for camera in self._disparity_cameras
        }

    @property
    def _topics(self) -> tuple[str, ...]:
        topics = [_TIMING_TOPIC, *self._index_topics.values()]
        for side in self._sides:
            topics += self._side_topics(side)
            if self._include_imu:
                topics += [
                    _topic(side, f"{segment}_imu.{field}")
                    for segment in IMU_SEGMENTS
                    for field in ("angular_velocity", "proper_acceleration")
                ]
            if self._include_status_flags:
                topics += [
                    _topic(side, f"{sensor}.status_flags") for sensor in STATUS_SENSORS
                ]

        return tuple(topics)

    # -------------------------------------------------------------------- read

    def _read(self, path: Path) -> tuple[_Floats, _Ints, dict[str, dict[str, str]]]:
        floats: _Floats = defaultdict(list)
        ints: _Ints = defaultdict(list)
        metadata: dict[str, dict[str, str]] = {}

        with path.open("rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            # §17.1: per-episode declarations. Presence is declared, never
            # inferred from topic absence.
            metadata = {
                record.name: record.metadata for record in reader.iter_metadata()
            }

            f.seek(0)
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, channel, _message, decoded in reader.iter_decoded_messages(
                topics=self._topics
            ):
                topic = channel.topic
                match topic.rsplit(".", 1)[-1]:
                    case "transform":
                        floats[topic].append([
                            decoded.x,
                            decoded.y,
                            decoded.z,
                            decoded.qx,
                            decoded.qy,
                            decoded.qz,
                            decoded.qw,
                        ])

                    case "orientation":
                        floats[topic].append([
                            decoded.qx,
                            decoded.qy,
                            decoded.qz,
                            decoded.qw,
                        ])

                    case "angular_velocity" | "proper_acceleration":
                        floats[topic].append([decoded.x, decoded.y, decoded.z])

                    case "status_flags":
                        ints[topic].append([decoded.bit_mapped_flags])

                    case "rgmp":
                        ints[topic].append([
                            decoded.device_timestamp_us * 1_000,
                            decoded.host_arrival_time_ns,
                        ])

                    case _:  # observation.images.{camera}
                        ints[topic].append([
                            decoded.frame_index,
                            decoded.device_timestamp_ns,
                            decoded.host_arrival_time_ns,
                        ])

        return floats, ints, metadata

    def _disparity_declarations(
        self, metadata: dict[str, dict[str, str]], calibration: NeroArmsCalibration
    ) -> dict[str, DisparityDeclaration]:
        """§21.3/§21.4: the declared stereo mode, asserted, never inferred.

        A missing record when disparity ingestion is switched on is a hard
        failure -- falling back to the calibration block would be exactly the
        §17.1 mistake of inferring presence from absence.
        """
        declarations: dict[str, DisparityDeclaration] = {}
        for camera in self._disparity_cameras:
            name = f"{DISPARITY_METADATA_PREFIX}{camera}"
            if (record := metadata.get(name)) is None:
                logger.error(
                    msg := "§21.3: episode metadata does not declare the "
                    "disparity stream; `max_disparity`/`subpixel`/"
                    "`extended_disparity` must be declared, never inferred",
                    record=name,
                    available=sorted(metadata),
                )

                raise ValueError(msg)

            declaration = DisparityDeclaration.from_mcap_metadata(record)
            if (stereo := calibration.stereo.get(camera)) is not None:
                declaration.check_against(stereo, camera=camera)

            declarations[camera] = declaration

        return declarations

    # ---------------------------------------------------------------- timebase

    def _glove(self, floats: _Floats, ints: _Ints) -> _Stream:
        if _TIMING_TOPIC not in ints:
            logger.error(msg := "missing timing topic", topic=_TIMING_TOPIC)

            raise ValueError(msg)

        timing = np.asarray(ints[_TIMING_TOPIC], dtype=np.int64)
        glove = _Stream(timing[:, 0], timing[:, 1], self._offset_percentile)

        # §3.1: the glove sensor topics carry no timestamps of their own; they
        # are published one-per-`timing.rgmp` sample, so the join is ordinal --
        # a single length mismatch would silently shift every pose.
        for topic, values in (*floats.items(), *ints.items()):
            if (
                topic.startswith((_IMAGE_TOPIC_PREFIX, DISPARITY_TOPIC_PREFIX))
                or topic == _TIMING_TOPIC
            ):
                continue

            if len(values) != len(glove.device_ns):
                logger.error(
                    msg := "glove topic length != `timing.rgmp` length",
                    topic=topic,
                    length=len(values),
                    expected=len(glove.device_ns),
                )

                raise ValueError(msg)

        return glove

    def _cameras_from(self, ints: _Ints) -> dict[str, _Camera]:
        cameras: dict[str, _Camera] = {}
        for camera, topic in self._index_topics.items():
            if topic not in ints:
                logger.error(msg := "missing camera topic", topic=topic)

                raise ValueError(msg)

            values = np.asarray(ints[topic], dtype=np.int64)
            cameras[camera] = _Camera(
                values[:, 0].astype(np.int32),
                _Stream(values[:, 1], values[:, 2], self._offset_percentile),
            )

        return cameras

    def _resample(self, glove: _Stream, reference: _Camera) -> _Resampling:
        glove_ns = glove.common_ns
        grid_ns = reference.stream.common_ns

        # §3.3: never extrapolate. Reference frames outside the glove stream's
        # span are dropped rather than clamped.
        inside = (grid_ns >= glove_ns[0]) & (grid_ns <= glove_ns[-1])
        if (dropped := int((~inside).sum())) > 0:
            logger.debug(
                "dropped reference frames outside the glove stream span",
                count=dropped,
                total=len(grid_ns),
            )

        grid_ns = grid_ns[inside]
        if grid_ns.size == 0:
            logger.error(msg := "no reference frames overlap the glove stream")

            raise ValueError(msg)

        hi = np.searchsorted(glove_ns, grid_ns, side="left").clip(1, len(glove_ns) - 1)
        lo = hi - 1
        t_lo, t_hi = glove_ns[lo], glove_ns[hi]
        to_lo, to_hi = grid_ns - t_lo, t_hi - grid_ns

        # §3.4: the actual interpolation distance -- not the clock-offset drift,
        # which is near-constant and would make the tolerance check vacuous.
        residual_ms = np.minimum(to_lo, to_hi).astype(np.float64) / _NS_PER_MS
        if (worst := float(residual_ms.max())) > self._align_tolerance_ms:
            logger.error(
                msg := "alignment residual exceeds one glove period",
                residual_ms=round(worst, 3),
                tolerance_ms=self._align_tolerance_ms,
            )

            raise ValueError(msg)

        return _Resampling(
            grid_ns=grid_ns,
            frame_index=reference.frame_index[inside],
            lo=lo,
            hi=hi,
            u=(to_lo / np.maximum(t_hi - t_lo, 1)),
            nearest=np.where(to_lo <= to_hi, lo, hi),
            residual_ms=residual_ms,
        )

    # ------------------------------------------------------------------- state

    def _side_blocks(  # noqa: C901
        self, floats: _Floats, ints: _Ints, r: _Resampling
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.bool_],
    ]:
        """§6.1: bimanual state blocks plus the `side_valid` mask."""
        n, n_sides = len(r.grid_ns), len(SIDES)
        state = np.zeros((n, n_sides, STATE_DIM_QUAT), dtype=np.float64)
        imu = np.zeros((n, n_sides, IMU_DIM), dtype=np.float64)
        status = np.zeros((n, n_sides, len(STATUS_SENSORS)), dtype=np.int32)
        side_valid = np.zeros(n_sides, dtype=bool)

        for side in self._sides_present(floats):
            s = SIDES.index(side)
            side_valid[s] = True
            offset = 0
            for topic in self._side_topics(side):
                values = np.asarray(floats[topic], dtype=np.float64)
                match values.shape[-1]:
                    case 7:  # §3.3: linear for translations, SLERP for rotations
                        state[:, s, offset : offset + _POSE_WIDTH] = np.concatenate(
                            [r.lerp(values[:, :3]), r.slerp(values[:, 3:])], axis=-1
                        )
                        offset += _POSE_WIDTH

                    case 4:  # rotation only
                        state[:, s, offset : offset + _QUAT_WIDTH] = r.slerp(values)
                        offset += _QUAT_WIDTH

                    case _:
                        raise RuntimeError

            if self._include_imu:
                for i, segment in enumerate(IMU_SEGMENTS):
                    fields = ("angular_velocity", "proper_acceleration")
                    for j, field in enumerate(fields):
                        k = (i * len(fields) + j) * 3
                        imu[:, s, k : k + 3] = r.lerp(
                            np.asarray(
                                floats[_topic(side, f"{segment}_imu.{field}")],
                                dtype=np.float64,
                            )
                        )

            if self._include_status_flags:
                for i, sensor in enumerate(STATUS_SENSORS):
                    # §2.5: carried through untouched; never interpolated
                    status[:, s, i] = np.asarray(
                        ints[_topic(side, f"{sensor}.status_flags")], dtype=np.int32
                    )[r.nearest, 0]

        if not side_valid.any():
            logger.error(msg := "no sides present in recording")

            raise ValueError(msg)

        return state, imu, status, side_valid

    def _sides_present(self, floats: _Floats) -> tuple[str, ...]:
        present = []
        for side in self._sides:
            if all(topic in floats for topic in self._side_topics(side)):
                present.append(side)
            else:
                # §6.1: a missing side is zeros + mask, never a dropped column
                logger.debug("side absent from recording; masking out", side=side)

        return tuple(present)

    @staticmethod
    def _relative_to_start(
        state: npt.NDArray[np.float64], side_valid: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float64]:
        """§4: arm pose and hub orientation relative to the episode's first sample.

        Finger poses are already expressed relative to the glove hub and are left
        untouched.
        """
        out = state.copy()
        for s in np.flatnonzero(side_valid):
            r0 = quat_to_matrix(state[0, s, 3:7])
            out[:, s, :3] = (state[:, s, :3] - state[0, s, :3]) @ r0
            out[:, s, 3:7] = _quat_relative(state[0, s, 3:7], state[:, s, 3:7])
            out[:, s, -4:] = _quat_relative(state[0, s, -4:], state[:, s, -4:])

        return out

    # ---------------------------------------------------------- camera columns

    def _frame_index_columns(
        self, cameras: dict[str, _Camera], r: _Resampling, n: int
    ) -> dict[str, pl.Series]:
        """Join every indexed stream on its own `frame_index` (§3, §21.4).

        Covers the mp4 cameras and the disparity mkv identically -- the latter is
        keyed `disparity.{camera}` and is not assumed to share a frame count with
        anything.
        """
        grid_ns = r.grid_ns[:n]
        columns: dict[str, pl.Series] = {}
        for camera, (frame_index, stream) in cameras.items():
            name = f"frame_index.{camera}"
            if camera == self._reference_camera:
                columns[name] = pl.Series(name, r.frame_index[:n], pl.Int32)

                continue

            other_ns = stream.common_ns
            j = np.searchsorted(other_ns, grid_ns, side="left").clip(
                0, len(other_ns) - 1
            )
            j_prev = (j - 1).clip(0, len(other_ns) - 1)
            j = np.where(
                np.abs(other_ns[j_prev] - grid_ns) < np.abs(other_ns[j] - grid_ns),
                j_prev,
                j,
            )
            gap_ms = np.abs(other_ns[j] - grid_ns) / _NS_PER_MS
            if (beyond := int((gap_ms > self._camera_gap_tolerance_ms).sum())) > 0:
                logger.warning(
                    "camera frames beyond gap tolerance",
                    camera=camera,
                    count=beyond,
                    max_gap_ms=round(float(gap_ms.max()), 3),
                )

            columns[name] = pl.Series(name, frame_index[j], pl.Int32)

        return columns

    # ------------------------------------------------------------------- build

    def _build(self, path: Path, calibration_path: Path) -> pl.DataFrame:  # noqa: PLR0914
        calibration = NeroArmsCalibration.from_path(calibration_path)
        floats, ints, metadata = self._read(path)
        declarations = self._disparity_declarations(metadata, calibration)
        if declarations:
            logger.debug(
                "disparity streams declared",
                declarations={
                    camera: declaration.model_dump(exclude_none=True)
                    for camera, declaration in declarations.items()
                },
            )

        glove = self._glove(floats, ints)
        cameras = self._cameras_from(ints)
        resampling = self._resample(glove, cameras[self._reference_camera])

        logger.debug(
            "clock offsets",
            offset_ns={"glove": glove.offset_ns}
            | {camera: c.stream.offset_ns for camera, c in cameras.items()},
            latency_spread_ms={"glove": round(glove.latency_spread_ms, 3)}
            | {
                camera: round(c.stream.latency_spread_ms, 3)
                for camera, c in cameras.items()
            },
            offset_drift_ms={"glove": round(glove.offset_drift_ms, 3)}
            | {
                camera: round(c.stream.offset_drift_ms, 3)
                for camera, c in cameras.items()
            },
        )

        state, imu, status, side_valid = self._side_blocks(floats, ints, resampling)

        # §6.2: action(t) == states [t+1 .. t+H]; rows without a full chunk are
        # dropped rather than clamped or wrapped.
        h = self._action_horizon
        if (n := len(resampling.grid_ns) - h) <= 0:
            logger.error(
                msg := "episode shorter than the action horizon",
                length=len(resampling.grid_ns),
                action_horizon=h,
            )

            raise ValueError(msg)

        future = np.stack([state[i + 1 : i + 1 + h] for i in range(n)])

        # §9: goal is the final arm position / final frame of the episode
        goal_xyz = np.zeros((len(SIDES), 3), dtype=np.float64)
        goal_xyz[side_valid] = state[-1, side_valid, :3]

        columns = (
            self._frame_index_columns(cameras, resampling, n)
            | {
                "state.pose": _array("state.pose", state[:n], pl.Float32()),
                "state.pose_rel_start": _array(
                    "state.pose_rel_start",
                    self._relative_to_start(state, side_valid)[:n],
                    pl.Float32(),
                ),
                "action.future_state": _array(
                    "action.future_state", future, pl.Float32()
                ),
                "side_valid": _array(
                    "side_valid", _repeat(side_valid, n), pl.Boolean()
                ),
                "align_residual_ms": pl.Series(
                    "align_residual_ms", resampling.residual_ms[:n], pl.Float32
                ),
                "camera_cond": _array(
                    "camera_cond",
                    _repeat(calibration.cond(self._cameras), n),
                    pl.Float32(),
                ),
                "camera_cond.placeholder": pl.Series(
                    "camera_cond.placeholder",
                    np.full(n, calibration.placeholder),
                    pl.Boolean,
                ),
                "goal.xyz": _array("goal.xyz", _repeat(goal_xyz, n), pl.Float32()),
            }
            | {
                f"goal.frame_index.{camera}": pl.Series(
                    f"goal.frame_index.{camera}",
                    np.full(n, int(camera_.frame_index[-1])),
                    pl.Int32,
                )
                for camera, camera_ in cameras.items()
            }
        )

        if self._include_imu:
            columns["state.imu"] = _array("state.imu", imu[:n], pl.Float32())

        if self._include_status_flags:
            columns["state.status_flags"] = _array(
                "state.status_flags", status[:n], pl.Int32()
            )

        return pl.DataFrame(columns)


def _quat_relative(
    q0: npt.NDArray[np.float64], q: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """`q0^-1 * q`, canonicalised."""
    x0, y0, z0, w0 = -q0[0], -q0[1], -q0[2], q0[3]
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return canonicalize_quat(
        np.stack(
            [
                w0 * x + x0 * w + y0 * z - z0 * y,
                w0 * y - x0 * z + y0 * w + z0 * x,
                w0 * z + x0 * y - y0 * x + z0 * w,
                w0 * w - x0 * x - y0 * y - z0 * z,
            ],
            axis=-1,
        )
    )


def _repeat(values: npt.NDArray[np.generic], n: int) -> npt.NDArray[np.generic]:
    """Broadcast an episode-constant block to one row per sample."""
    return np.broadcast_to(values, (n, *values.shape))


def _nested(dtype: pl.DataType, shape: tuple[int, ...]) -> pl.DataType:
    for size in reversed(shape):
        dtype = pl.Array(dtype, size)

    return dtype


def _array(name: str, values: npt.NDArray[np.generic], dtype: pl.DataType) -> pl.Series:
    return pl.Series(name, values.tolist(), _nested(dtype, values.shape[1:]))
