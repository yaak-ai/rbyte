from collections.abc import Callable, Sequence
from typing import override

import polars as pl
from cachetools import LRUCache, cachedmethod
from google.protobuf.descriptor import Descriptor
from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding
from mcap_protobuf.decoder import DecoderFactory as McapProtobufDecoderFactory
from ptars import HandlerPool


class ProtobufDecoderFactory(McapDecoderFactory):
    def __init__(self) -> None:
        self._message_descriptor_cache: LRUCache[tuple[str, bytes], Descriptor] = (
            LRUCache(maxsize=32)
        )

    @override
    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes | Sequence[bytes]], pl.DataFrame] | None:
        if (
            message_encoding == MessageEncoding.Protobuf
            and schema is not None
            and schema.encoding == SchemaEncoding.Protobuf
        ):
            descriptor = self._get_message_descriptor(schema)
            handler = HandlerPool([descriptor.file]).get_for_message(descriptor)

            def decoder(data: bytes | Sequence[bytes]) -> pl.DataFrame:
                record_batch = handler.list_to_record_batch(
                    [data] if isinstance(data, bytes) else data
                )
                return pl.from_arrow(record_batch, rechunk=False)  # ty: ignore[invalid-return-type]

            return decoder

        return None

    @cachedmethod(
        cache=lambda self: self._message_descriptor_cache,
        key=lambda _, schema: (schema.name, schema.data),
    )
    def _get_message_descriptor(self, schema: Schema) -> Descriptor:  # ruff:ignore[no-self-use]
        decoder = McapProtobufDecoderFactory().decoder_for(
            MessageEncoding.Protobuf, schema
        )

        return decoder(b"").DESCRIPTOR  # ty:ignore[call-non-callable]
