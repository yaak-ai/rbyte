from collections.abc import Callable
from operator import attrgetter
from typing import override

import more_itertools as mit
import polars as pl
from cachetools import cached
from google.protobuf.descriptor_pb2 import FileDescriptorProto, FileDescriptorSet
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.message import Message
from google.protobuf.message_factory import GetMessageClassesForFiles
from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.exceptions import McapError
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding
from ptars import HandlerPool
from structlog import get_logger

logger = get_logger(__name__)


class ProtobufDecoderFactory(McapDecoderFactory):
    def __init__(self) -> None:
        self._handler_pool: HandlerPool = HandlerPool()

    @override
    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes], pl.DataFrame] | None:
        if (
            message_encoding == MessageEncoding.Protobuf
            and schema is not None
            and schema.encoding == SchemaEncoding.Protobuf
        ):
            message_type = self._get_message_type(schema)
            handler = self._handler_pool.get_for_message(message_type.DESCRIPTOR)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

            def decoder(data: bytes) -> pl.DataFrame:
                record_batch = handler.list_to_record_batch([data])  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                return pl.from_arrow(record_batch, rechunk=False)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportReturnType]

            return decoder

        return None

    @staticmethod
    @cached(cache={}, key=attrgetter("id"))
    def _get_message_type(schema: Schema) -> type[Message]:
        fds = FileDescriptorSet.FromString(schema.data)
        pool = DescriptorPool()
        descriptor_by_name = mit.map_reduce(
            fds.file, keyfunc=attrgetter("name"), reducefunc=mit.one
        )

        def _add(fd: FileDescriptorProto) -> None:
            for dependency in fd.dependency:
                if dependency in descriptor_by_name:
                    _add(descriptor_by_name.pop(dependency))

            pool.Add(fd)  # pyright: ignore[reportUnknownMemberType]

        while descriptor_by_name:
            _add(descriptor_by_name.popitem()[1])

        message_types = GetMessageClassesForFiles([fd.name for fd in fds.file], pool)

        if (message_type := message_types.get(schema.name, None)) is None:
            logger.error(msg := "FileDescriptorSet missing schema", schema=schema)

            raise McapError(msg)

        return message_type
