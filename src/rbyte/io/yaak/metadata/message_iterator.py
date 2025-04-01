import struct
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from mmap import mmap
from typing import BinaryIO, ClassVar, Self, override

from google.protobuf.message import Message
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars

from rbyte.io.yaak.proto import can_pb2, sensor_pb2

logger = get_logger(__name__)


def to_uint32(buf: bytes) -> int:
    return int(struct.unpack("I", buf)[0])


class YaakMetadataMessageIterator(Iterator[tuple[type[Message], bytes]]):
    """An iterator over a metadata file(-like object) producing messages."""

    MESSAGE_TYPES: ClassVar[dict[int, type[Message]]] = {
        0: sensor_pb2.Gnss,
        4: sensor_pb2.ImageMetadata,
        7: can_pb2.VehicleMotion,
        8: can_pb2.VehicleState,
    }

    FILE_HEADER_VERSION: int = 1
    FILE_HEADER_LEN: int = 12
    MESSAGE_HEADER_LEN: int = 8

    @validate_call
    def __init__(
        self,
        file: InstanceOf[BinaryIO] | InstanceOf[mmap],
        *,
        message_types: AbstractSet[type[Message]] | None = None,
    ) -> None:
        super().__init__()

        self._file: BinaryIO | mmap = file

        for expected_val, desc in (
            (self.FILE_HEADER_LEN, "file header length"),
            (self.FILE_HEADER_VERSION, "file header version"),
            (self.MESSAGE_HEADER_LEN, "message header length"),
        ):
            if (val := to_uint32(self._file.read(4))) != expected_val:
                msg = f"invalid {desc}: {val}, expected: {expected_val}"
                raise ValueError(msg)

        if message_types is None:
            self._message_types: dict[int, type[Message]] = self.MESSAGE_TYPES
        else:
            if unknown_message_types := (
                message_types - set(self.MESSAGE_TYPES.values())
            ):
                with bound_contextvars(unknown_message_types=unknown_message_types):
                    logger.error(msg := "unknown message types")
                    raise ValueError(msg)

            self._message_types = {
                k: v for k, v in self.MESSAGE_TYPES.items() if v in message_types
            }

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> tuple[type[Message], bytes]:
        msg_type = None
        while msg_type is None:
            try:
                msg_data = self._read_message()
            except Exception as exc:
                logger.warning("failed to read message", error=exc)
                raise StopIteration from exc
            else:
                if msg_data is None:
                    raise StopIteration

                msg_type_idx, msg_buf = msg_data
                msg_type = self._message_types.get(msg_type_idx)

        return msg_type, msg_buf  # pyright: ignore[reportPossiblyUnboundVariable]

    def _read_message(self) -> tuple[int, bytes] | None:
        msg_type_idx_buf = self._file.read(4)
        if not msg_type_idx_buf:
            return None

        msg_type_idx = to_uint32(msg_type_idx_buf)
        msg_len = to_uint32(self._file.read(4))
        msg_buf = self._file.read(msg_len)

        return (msg_type_idx, msg_buf)
