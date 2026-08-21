"""Decoding for the nero-arms disparity stream (data contract §21.16).

**The bit depth chooses the decoder, and that is the whole point of this module.**
Two measured findings (§21.2) fix the routing:

1. FFV1 8-bit `gray` is **bit-exact** through torchcodec, which rbyte already
   uses for the mp4s -- so an 8-bit stream reuses the `ImageFrameIndex` +
   `frame_index` machinery with no new decoder.
2. ⚠️ torchcodec **silently down-converts 16-bit video to uint8**. It does not
   raise; it returns plausible-looking wrong data (measured max error 58751).
   FFV1 `gray16le` round-trips bit-exact through **PyAV**.

Under the old 8-bit-only rule (§21.1) 16-bit was simply *refused* here. §21.16
dropped that restriction -- integer disparity cannot resolve the manipulated
object at the rig's working distance (§21.14) -- so the refusal becomes a
**routing rule** instead:

* 16-bit  -> `_PyAvDisparityDecoder`   (the `pyav` extra)
* 8-bit   -> `_TorchCodecDisparityDecoder` (the `video` extra)

and `_TorchCodecDisparityDecoder` **refuses 16-bit itself**, independently of the
routing that already keeps it away. That redundancy is deliberate: the routing is
the mechanism, but torchcodec's failure on 16-bit is *silent*, so the path that
could commit it must be closed at the point of use as well as upstream of it.
Nothing here ever hands a 16-bit stream to torchcodec, including for probing --
the container is probed with PyAV whenever PyAV is installed.

A lossy codec is refused for the same family of reasons: libx264 on 8-bit
disparity gave a max error of 81 levels at 95 levels full scale.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import suppress
from typing import Any, final, override

import numpy as np
import torch
from pydantic import FilePath, InstanceOf, validate_call
from structlog import get_logger
from torch import Tensor

from rbyte.samples.nero.calibration import NeroArmsCalibration, StereoCalibration
from rbyte.samples.nero.disparity import (
    BIT_DEPTH_PYAV_ONLY,
    DISPARITY_PIXEL_FORMATS,
    LOSSLESS_CODECS,
    PIXEL_FORMAT,
    DisparityDeclaration,
    DisparityOutput,
    check_store_against_mode,
    disparity_to_depth,
)
from rbyte.streams.base import StreamSource

__all__ = ["NeroArmsDisparitySource"]

logger = get_logger(__name__)


def _probe(source: str, stream_index: int | None = None) -> tuple[str, str]:
    """`(codec, pixel_format)` of the video stream, without decoding a frame.

    PyAV first, ALWAYS, when it is installed: probing a 16-bit file through
    torchcodec would mean constructing the decoder this module exists to keep
    away from such a file. Only when PyAV is absent does the probe fall back to
    torchcodec's metadata -- and then a 16-bit answer is a hard failure telling
    the caller to install the `pyav` extra, because there is no legal decoder for
    it in that environment.
    """
    try:
        import av  # ruff:ignore[import-outside-top-level]
    except ImportError:
        pass
    else:
        with av.open(source) as container:
            stream = (
                container.streams.video[0]
                if stream_index is None
                else container.streams[stream_index]
            )

            return stream.codec_context.name, stream.format.name  # ty:ignore[unresolved-attribute]

    import torchcodec.decoders  # ruff:ignore[import-outside-top-level]

    metadata = torchcodec.decoders.VideoDecoder(
        source=source, stream_index=stream_index, seek_mode="approximate"
    ).metadata
    codec, pixel_format = metadata.codec or "", metadata.pixel_format or ""
    if DISPARITY_PIXEL_FORMATS.get(pixel_format) == BIT_DEPTH_PYAV_ONLY:
        logger.error(
            msg := "§21.16: this is a 16-bit disparity stream and PyAV is not "
            "installed. torchcodec would return uint8 for it WITHOUT raising, so "
            "there is no decoder here that can read it -- install the `pyav` "
            "extra.",
            source=source,
            pixel_format=pixel_format,
        )

        raise RuntimeError(msg)

    return codec, pixel_format


class _DisparityDecoder(ABC):
    """`(N, 1, H, W)` stored disparity by frame ordinal, and nothing else."""

    @property
    @abstractmethod
    def num_frames(self) -> int: ...

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        """`(width, height)`."""

    @abstractmethod
    def frames(self, indices: Sequence[int]) -> Tensor: ...


@final
class _TorchCodecDisparityDecoder(_DisparityDecoder):
    """8-bit `gray` only. **Refuses 16-bit rather than accepting it.**

    The refusal is the load-bearing part of this class. torchcodec returns uint8
    for a 16-bit stream and does not raise, so accepting one here would produce a
    depth map that is wrong by up to 58751 and looks entirely ordinary. Every
    other failure mode in rbyte announces itself; this one does not, so it is
    closed here even though `NeroArmsDisparitySource` already routes 16-bit
    to PyAV and never constructs this class for it.
    """

    def __init__(
        self,
        source: str,
        *,
        stream_index: int | None = None,
        num_ffmpeg_threads: int = 1,
        device: str | None = None,
    ) -> None:
        import torchcodec.decoders  # ruff:ignore[import-outside-top-level]

        self._decoder = torchcodec.decoders.VideoDecoder(
            source=source,
            stream_index=stream_index,
            dimension_order="NCHW",
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=device,
            seek_mode="exact",
        )
        self._check_eight_bit(source)
        self._check_channels()

    def _check_eight_bit(self, source: str) -> None:
        pixel_format = self._decoder.metadata.pixel_format
        if pixel_format != PIXEL_FORMAT:
            logger.error(
                msg := "§21.16: torchcodec decodes 8-bit `gray` ONLY. It SILENTLY "
                "down-converts 16-bit video to uint8 -- it does not raise, it "
                "returns wrong data (measured max error 58751). A 16-bit "
                "disparity stream must be decoded with PyAV.",
                source=source,
                pixel_format=pixel_format,
                expected=PIXEL_FORMAT,
            )

            raise ValueError(msg)

    def _check_channels(self) -> None:
        """torchcodec hands `gray` back NCHW with the plane replicated across
        three channels. Assert that once, here, rather than on every decode: at
        the measured 1280x800 an all-close over a clip is tens of MB of pointless
        traffic per sample, and the behaviour is fixed for a given file.
        """
        if not self._decoder.metadata.num_frames:
            return

        frames = self._decoder.get_frame_at(index=0).data
        if frames.shape[-3] != 1 and not bool((frames == frames[..., :1, :, :]).all()):
            logger.error(
                msg := "disparity frames are not single-channel gray",
                shape=tuple(frames.shape),
            )

            raise ValueError(msg)

    @property
    @override
    def num_frames(self) -> int:
        return self._decoder.metadata.num_frames or 0

    @property
    @override
    def size(self) -> tuple[int, int]:
        metadata = self._decoder.metadata

        return (metadata.width or 0, metadata.height or 0)

    @override
    def frames(self, indices: Sequence[int]) -> Tensor:
        data = self._decoder.get_frames_at(indices=list(indices)).data

        # replication across the three channels is checked once, in `__init__`
        return data[..., :1, :, :].contiguous()


@final
class _PyAvDisparityDecoder(_DisparityDecoder):
    """`gray`/`gray16le` through PyAV, bit-exact -- the 16-bit path (§21.16).

    **Sequential, with a cursor.** PyAV has no frame-index API, and rather than
    guess at `seek()` semantics (whose units are stream time-base, not frames)
    this decodes in order and counts. FFV1 is written `-g 1`, every frame a
    keyframe, so a restart is cheap and correct; the cursor means a forward walk
    -- which is what a dataloader reading a clip does -- costs one pass, while a
    backwards jump reopens the container. Random access over a long file is
    therefore O(index), not O(1). That is a real cost, and it is the price of
    bit-exactness on this stream; nothing else reads 16-bit FFV1 correctly.

    16-bit frames come back as **int32**, not `torch.uint16`: torch has the dtype
    but not the operators (`gt_cpu` is unimplemented for UInt16), so `disparity >
    0` -- which is the §21.4 validity mask, i.e. the thing every consumer of this
    stream does first -- would raise. int32 holds every uint16 value exactly, so
    nothing is lost but memory.
    """

    def __init__(self, source: str, *, stream_index: int | None = None) -> None:
        import av  # ruff:ignore[import-outside-top-level]

        self._av = av
        self._source = source
        self._stream_index = stream_index
        self._container: Any = None
        self._frames: Iterator[Any] | None = None
        self._cursor = 0
        self._cached: tuple[int, np.ndarray] | None = None

        with av.open(source) as container:
            stream = self._select(container)
            self._size = (stream.width, stream.height)
            self._pixel_format = stream.format.name
            declared = stream.frames
        # `stream.frames` is 0 for FFV1 in MKV, so the count is established by
        # decoding once. Not free, but a length that is silently 0 would make
        # every window over this stream vanish from the dataset.
        self._num_frames = declared if declared > 0 else self._count()

    def _select(self, container: Any) -> Any:  # ruff:ignore[any-type]
        """PyAV's container/stream types are only importable lazily, hence `Any`."""
        if self._stream_index is None:
            return container.streams.video[0]

        return container.streams[self._stream_index]

    def _count(self) -> int:
        with self._av.open(self._source) as container:
            stream = self._select(container)

            return sum(1 for _ in container.decode(stream))

    def _reopen(self) -> None:
        self.close()
        self._container = self._av.open(self._source)
        self._frames = self._container.decode(self._select(self._container))
        self._cursor = 0

    def _frame_at(self, index: int) -> np.ndarray:
        if self._cached is not None and self._cached[0] == index:
            return self._cached[1]

        if self._frames is None or index < self._cursor:
            self._reopen()

        frames = self._frames
        if frames is None:
            msg = f"could not open {self._source} for decoding"

            raise RuntimeError(msg)

        array: np.ndarray | None = None
        while self._cursor <= index:
            try:
                frame = next(frames)
            except StopIteration:
                msg = (
                    f"disparity frame {index} is past the end of {self._source} "
                    f"({self._cursor} frames)"
                )

                raise IndexError(msg) from None

            array = frame.to_ndarray()
            self._cursor += 1

        if array is None:
            # `_reopen` puts the cursor at 0 and a repeat is served from
            # `_cached`, so the loop always runs at least once. Stated rather
            # than asserted, because a silent None here would become a shape
            # error three frames later.
            msg = f"disparity frame {index} was not produced by {self._source}"

            raise RuntimeError(msg)

        self._cached = (index, array)

        return array

    @property
    @override
    def num_frames(self) -> int:
        return self._num_frames

    @property
    @override
    def size(self) -> tuple[int, int]:
        return self._size

    @override
    def frames(self, indices: Sequence[int]) -> Tensor:
        # ascending order so a batch costs one forward walk rather than one
        # reopen per out-of-order index; the caller's order is restored after
        order = sorted(range(len(indices)), key=indices.__getitem__)
        decoded: list[np.ndarray | None] = [None] * len(indices)
        for position in order:
            decoded[position] = self._frame_at(indices[position])

        stacked = np.stack([array for array in decoded if array is not None])
        if stacked.dtype == np.uint16:
            # torch.uint16 exists but `>` is not implemented for it, and `> 0` is
            # the §21.4 validity mask every consumer starts from.
            stacked = stacked.astype(np.int32)

        return torch.from_numpy(stacked).unsqueeze(-3).contiguous()

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = None
        self._frames = None
        self._cached = None

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


@final
class NeroArmsDisparitySource(StreamSource[int]):
    """Decode `{camera}_disparity.mkv` and optionally convert it to metric depth.

    One source == one output (`disparity` / `depth` / `valid`), because an rbyte
    stream is a single tensor. A depth-plus-mask config declares two streams over
    the same file.

    The decoder is chosen from the container's pixel format (§21.16): 16-bit goes
    to PyAV, 8-bit to torchcodec. A 16-bit stream additionally requires a
    DECLARED subpixel mode -- from `declaration` or from `stereo` -- because its
    values are `disparity * 2**fractional_bits` and without the exponent they are
    not a physical quantity at all. That is refused, not defaulted.

    Deliberately exposes **no** transforms: resizing a disparity map scales
    `fx` but not the stored disparity values, so `fx * baseline / disparity`
    would come out wrong by the resize factor (see `StereoCalibration`).
    """

    @validate_call
    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        source: FilePath | str,
        output: DisparityOutput = DisparityOutput.disparity,
        camera: str | None = None,
        calibration_path: FilePath | None = None,
        stereo: InstanceOf[StereoCalibration] | None = None,
        declaration: InstanceOf[DisparityDeclaration] | None = None,
        stream_index: int | None = None,
        num_ffmpeg_threads: int = 1,
        device: str | None = None,
    ) -> None:
        super().__init__()

        self._output = output
        path = str(source)
        codec, pixel_format = _probe(path, stream_index)
        self._check_lossless(path, codec)
        self._bit_depth = self._check_pixel_format(path, pixel_format)
        self._stereo = (
            None
            if output is not DisparityOutput.depth
            else self._resolve_stereo(camera, calibration_path, stereo)
        )
        self._check_store(
            path,
            declaration
            or self._stereo
            or self._optional_stereo(camera, calibration_path, stereo),
        )
        self._decoder: _DisparityDecoder = (
            _PyAvDisparityDecoder(path, stream_index=stream_index)
            if self._bit_depth == BIT_DEPTH_PYAV_ONLY
            else _TorchCodecDisparityDecoder(
                path,
                stream_index=stream_index,
                num_ffmpeg_threads=num_ffmpeg_threads,
                device=device,
            )
        )
        if self._stereo is not None:
            self._check_size(self._stereo)

    # -------------------------------------------------------------- validation

    @staticmethod
    def _check_lossless(source: str, codec: str) -> None:
        if codec not in LOSSLESS_CODECS:
            logger.error(
                msg := "§21.2: disparity stream is not losslessly coded. A lossy "
                "codec is destruction, not degradation -- libx264 on 8-bit "
                "disparity measured a max error of 81 levels at 95 full scale.",
                source=source,
                codec=codec,
                expected=sorted(LOSSLESS_CODECS),
            )

            raise ValueError(msg)

    @staticmethod
    def _check_pixel_format(source: str, pixel_format: str) -> int:
        """8 or 16, or a loud failure -- this is the routing decision."""
        if (bit_depth := DISPARITY_PIXEL_FORMATS.get(pixel_format)) is None:
            logger.error(
                msg := "§21.16: disparity is stored as `gray` or `gray16le`; "
                "anything else is not this stream",
                source=source,
                pixel_format=pixel_format,
                expected=sorted(DISPARITY_PIXEL_FORMATS),
            )

            raise ValueError(msg)

        return bit_depth

    def _check_store(
        self, source: str, mode: DisparityDeclaration | StereoCalibration | None
    ) -> None:
        """A 16-bit stream is uninterpretable without a DECLARED subpixel mode.

        The stored values are `disparity * 2**fractional_bits`. Absent the
        exponent there is no way to recover disparity, and therefore none to
        recover metres -- and it cannot be inferred from the pixels, since a
        stream at 3 bits and the same stream at 5 differ only by a factor the
        data does not carry. So this refuses rather than assuming §21.16's
        5 (which is merely depthai's default on a *fresh* config; a preset-built
        node was measured at 3).

        8-bit needs no declaration to be read raw, which is why it is exempt: the
        values are integer disparity levels whatever else is or is not declared.

        `DisparityOutput.valid` is exempt too, at any bit depth, and for the same
        reason turned around: it emits `disparity > 0`, which is the §21.4
        invalid test and is invariant under the scale. Nothing about it can be
        wrong by a power of two, so requiring the declaration would only block
        the legitimate depth-plus-mask config -- two streams over one file, where
        the mask half carries no calibration.
        """
        if mode is None:
            if (
                self._bit_depth == BIT_DEPTH_PYAV_ONLY
                and self._output is not DisparityOutput.valid
            ):
                logger.error(
                    msg := "§21.16: this is a 16-bit disparity stream and no "
                    "subpixel mode was declared. Its values are disparity * "
                    "2**fractional_bits and the exponent is not in the pixels, "
                    "so they cannot be interpreted -- pass `declaration` (from "
                    "the episode's `disparity.{camera}` record) or `stereo`.",
                    source=source,
                )

                raise ValueError(msg)

            return

        check_store_against_mode(
            observed_bit_depth=self._bit_depth,
            subpixel=mode.subpixel,
            subpixel_fractional_bits=mode.subpixel_fractional_bits,
            declared_bit_depth=getattr(mode, "bit_depth", None),
            source=source,
        )

    @staticmethod
    def _optional_stereo(
        camera: str | None,
        calibration_path: FilePath | None,
        stereo: StereoCalibration | None,
    ) -> StereoCalibration | None:
        """The declared mode for a NON-depth output, if one happens to be to hand.

        Raw `disparity` does not need calibration -- 8-bit levels are readable
        without it -- but a 16-bit stream needs the *subpixel* half of it, and the
        shipped config already passes `camera` + `calibration_path` for every
        output. Reading it here means that config works on a 16-bit stream instead
        of failing for a reason the user has already answered.

        Absence and placeholders are swallowed on purpose: for this output they
        are not errors, they merely supply no mode. If the stream then turns out
        to be 16-bit, `_check_store` fails on *that* -- the real reason -- rather
        than on a calibration this output never required.
        """
        if stereo is not None and not stereo.placeholder:
            return stereo

        if calibration_path is None or camera is None:
            return None

        try:
            return NeroArmsCalibration.from_path(calibration_path).stereo_for(camera)
        except (ValueError, OSError):
            return None

    @staticmethod
    def _resolve_stereo(
        camera: str | None,
        calibration_path: FilePath | None,
        stereo: StereoCalibration | None,
    ) -> StereoCalibration:
        if stereo is None:
            if calibration_path is None:
                logger.error(
                    msg := "§21.4: metric depth needs `stereo` or "
                    "`calibration_path`; pass `output=disparity` to ingest raw "
                    "disparity without calibration"
                )

                raise ValueError(msg)

            if camera is None:
                logger.error(msg := "`camera` is required with `calibration_path`")

                raise ValueError(msg)

            stereo = NeroArmsCalibration.from_path(calibration_path).stereo_for(camera)

        elif stereo.placeholder:
            logger.error(msg := "§21.4: `stereo` calibration is still PLACEHOLDER")

            raise ValueError(msg)

        return stereo

    def _check_size(self, stereo: StereoCalibration) -> None:
        if (size := self._decoder.size) != stereo.image_size:
            logger.error(
                msg := "disparity frame size != `stereo.image_size`; `fx_mono` "
                "describes a different resolution, so every depth would be "
                "scaled wrong",
                size=size,
                image_size=stereo.image_size,
            )

            raise ValueError(msg)

    # ------------------------------------------------------------------ decode

    def _disparity(self, indexes: int | Sequence[int]) -> Tensor:
        """`(N, 1, H, W)` stored disparity, bit-exact (§21.2).

        uint8 for an 8-bit stream, int32 for a 16-bit one -- see
        `_PyAvDisparityDecoder` for why not `torch.uint16`.
        """
        match indexes:
            case Sequence():
                return self._decoder.frames(list(indexes))

            case int():
                return self._decoder.frames([indexes]).squeeze(0)

    @override
    def __getitem__(self, indexes: int | Sequence[int]) -> Tensor:
        disparity = self._disparity(indexes)
        match self._output:
            case DisparityOutput.disparity:
                return disparity

            case DisparityOutput.valid:
                return disparity > 0

            case DisparityOutput.depth:
                if self._stereo is None:
                    raise RuntimeError

                return disparity_to_depth(disparity, self._stereo)[0]

    def __len__(self) -> int:
        return self._decoder.num_frames
