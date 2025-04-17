from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from mmap import ACCESS_READ, mmap
from operator import itemgetter
from typing import IO, final, override

import more_itertools as mit
import numpy.typing as npt
import torch
from mcap.data_stream import ReadDataStream
from mcap.decoder import DecoderFactory
from mcap.opcode import Opcode
from mcap.reader import SeekingReader
from mcap.records import Channel, Chunk, ChunkIndex, Message
from mcap.stream_reader import get_chunk_data_stream
from pydantic import FilePath, ImportString, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from torch import Tensor

from rbyte.config.base import BaseModel
from rbyte.io.base import TensorSource

logger = get_logger(__name__)


@dataclass(frozen=True)
class MessageIndex:
    chunk_start_offset: int
    message_start_offset: int
    message_length: int


@final
class McapTensorSource(TensorSource[int]):
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self,
        path: FilePath,
        topic: str,
        decoder_factory: ImportString[type[DecoderFactory]],
        decoder: Callable[[bytes], npt.ArrayLike],
        validate_crcs: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        super().__init__()

        with bound_contextvars(
            path=path.as_posix(), topic=topic, message_decoder_factory=decoder_factory
        ):
            self._path = path
            self._validate_crcs = validate_crcs

            summary = SeekingReader(
                stream=self._file, validate_crcs=self._validate_crcs
            ).get_summary()

            if summary is None:
                logger.error(msg := "missing summary")
                raise ValueError(msg)

            self._channel: Channel = mit.one(
                channel
                for channel in summary.channels.values()
                if channel.topic == topic
            )

            message_decoder = decoder_factory().decoder_for(
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
            self._decoder = decoder
            self._mmap = None

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
    def __getitem__(self, indexes: int | Sequence[int]) -> Tensor:
        match indexes:
            case Sequence():
                arrays: dict[int, npt.ArrayLike] = {}
                message_indexes = (self._message_indexes[idx] for idx in indexes)
                indexes_by_chunk_start_offset = mit.map_reduce(
                    zip(indexes, message_indexes, strict=True),
                    keyfunc=lambda x: x[1].chunk_start_offset,
                )

                for chunk_start_offset, chunk_indexes in sorted(
                    indexes_by_chunk_start_offset.items(), key=itemgetter(0)
                ):
                    _ = self._file.seek(chunk_start_offset + 1 + 8)
                    chunk = Chunk.read(ReadDataStream(self._file))
                    stream, _ = get_chunk_data_stream(
                        chunk, validate_crc=self._validate_crcs
                    )
                    for index, message_index in sorted(
                        chunk_indexes, key=lambda x: x[1].message_start_offset
                    ):
                        stream.read(message_index.message_start_offset - stream.count)  # pyright: ignore[reportUnusedCallResult]
                        message = Message.read(stream, message_index.message_length)
                        decoded_message = self._message_decoder(message.data)
                        arrays[index] = self._decoder(decoded_message.data)

                tensors = [torch.from_numpy(arrays[idx]) for idx in indexes]  # pyright: ignore[reportUnknownMemberType]

                return torch.stack(tensors)

            case int():
                message_index = self._message_indexes[indexes]
                _ = self._file.seek(message_index.chunk_start_offset + 1 + 8)
                chunk = Chunk.read(ReadDataStream(self._file))
                stream, _ = get_chunk_data_stream(chunk, self._validate_crcs)
                _ = stream.read(message_index.message_start_offset - stream.count)
                message = Message.read(stream, length=message_index.message_length)
                decoded_message = self._message_decoder(message.data)
                array = self._decoder(decoded_message.data)

                return torch.from_numpy(array)  # pyright: ignore[reportUnknownMemberType]

    @override
    def __len__(self) -> int:
        return len(self._message_indexes)

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
