from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from mmap import ACCESS_READ, mmap
from typing import IO, override

import more_itertools as mit
import numpy.typing as npt
import torch
from jaxtyping import Shaped
from mcap.data_stream import ReadDataStream
from mcap.decoder import DecoderFactory
from mcap.opcode import Opcode
from mcap.reader import SeekingReader
from mcap.records import Chunk, ChunkIndex, Message
from mcap.stream_reader import get_chunk_data_stream
from pydantic import ConfigDict, FilePath, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from torch import Tensor

from rbyte.io.frame.base import FrameReader

logger = get_logger(__name__)


@dataclass(frozen=True)
class MessageIndex:
    chunk_start_offset: int
    message_start_offset: int
    message_length: int


class McapFrameReader(FrameReader):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        path: FilePath,
        topic: str,
        message_decoder_factory: DecoderFactory,
        frame_decoder: Callable[[bytes], npt.ArrayLike],
        validate_crcs: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        super().__init__()

        with bound_contextvars(
            path=path.as_posix(),
            topic=topic,
            message_decoder_factory=message_decoder_factory,
        ):
            self._path = path
            self._validate_crcs = validate_crcs

            summary = SeekingReader(
                stream=self._file, validate_crcs=self._validate_crcs
            ).get_summary()

            if summary is None:
                logger.error(msg := "missing summary")
                raise ValueError(msg)

            self._channel = mit.one(
                channel
                for channel in summary.channels.values()
                if channel.topic == topic
            )

            message_decoder = message_decoder_factory.decoder_for(
                message_encoding=self._channel.message_encoding,
                schema=summary.schemas[self._channel.schema_id],
            )

            if message_decoder is None:
                logger.error(msg := "missing message decoder")
                raise RuntimeError(msg)

            self._message_decoder = message_decoder
            self._chunk_indexes = tuple(
                chunk_index
                for chunk_index in summary.chunk_indexes
                if self._channel.id in chunk_index.message_index_offsets
            )
            self._frame_decoder = frame_decoder

    @property
    def _file(self) -> IO[bytes]:
        match getattr(self, "_mmap", None):
            case mmap(closed=False):
                pass

            case None | mmap(closed=True):
                with self._path.open("rb") as f:
                    self._mmap = mmap(fileno=f.fileno(), length=0, access=ACCESS_READ)

            case _:
                raise RuntimeError

        return self._mmap  # pyright: ignore[reportReturnType]

    @override
    def read(self, indexes: Iterable[int]) -> Shaped[Tensor, "b h w c"]:
        frames: Mapping[int, npt.ArrayLike] = {}

        message_indexes_by_chunk_start_offset: Mapping[
            int, Iterable[tuple[int, MessageIndex]]
        ] = mit.map_reduce(
            zip(indexes, (self._message_indexes[idx] for idx in indexes), strict=True),
            keyfunc=lambda x: x[1].chunk_start_offset,
        )

        for (
            chunk_start_offset,
            chunk_message_indexes,
        ) in message_indexes_by_chunk_start_offset.items():
            self._file.seek(chunk_start_offset + 1 + 8)  # pyright: ignore[reportUnusedCallResult]
            chunk = Chunk.read(ReadDataStream(self._file))
            stream, _ = get_chunk_data_stream(chunk, validate_crc=self._validate_crcs)
            for frame_index, message_index in sorted(
                chunk_message_indexes, key=lambda x: x[1].message_start_offset
            ):
                stream.read(message_index.message_start_offset - stream.count)  # pyright: ignore[reportUnusedCallResult]
                message = Message.read(stream, message_index.message_length)
                decoded_message = self._message_decoder(message.data)
                frames[frame_index] = self._frame_decoder(decoded_message.data)

        return torch.stack([torch.from_numpy(frames[idx]) for idx in indexes])  # pyright: ignore[reportUnknownMemberType]

    @override
    def get_available_indexes(self) -> Sequence[int]:
        return range(len(self._message_indexes))

    @cached_property
    def _message_indexes(self) -> Sequence[MessageIndex]:
        return tuple(
            self._build_message_indexes(
                self._file,
                chunk_indexes=self._chunk_indexes,
                channel_id=self._channel.id,
                validate_crc=self._validate_crcs,
            )
        )

    @staticmethod
    def _build_message_indexes(
        f: IO[bytes],
        *,
        chunk_indexes: Iterable[ChunkIndex],
        channel_id: int,
        validate_crc: bool,
    ) -> Iterable[MessageIndex]:
        for chunk_index in chunk_indexes:
            f.seek(chunk_index.chunk_start_offset + 1 + 8)  # pyright: ignore[reportUnusedCallResult]
            chunk = Chunk.read(ReadDataStream(f))
            stream, stream_length = get_chunk_data_stream(chunk, validate_crc)

            while stream.count < stream_length:
                opcode = stream.read1()
                length = stream.read8()
                match opcode:
                    case Opcode.MESSAGE:
                        if Message.read(stream, length).channel_id == channel_id:
                            yield MessageIndex(
                                chunk_start_offset=chunk_index.chunk_start_offset,
                                message_start_offset=stream.count - length,
                                message_length=length,
                            )

                    case _:
                        stream.read(length)  # pyright: ignore[reportUnusedCallResult]
