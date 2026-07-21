from collections.abc import Callable, Sequence
from typing import override

import polars as pl
from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from structlog import get_logger

logger = get_logger(__name__)


class JsonDecoderFactory(McapDecoderFactory):
    @override
    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes | Sequence[bytes]], pl.DataFrame] | None:
        if (
            message_encoding == "json"
            and schema is not None
            and schema.encoding == "jsonschema"
        ):

            def decoder(data: bytes | Sequence[bytes]) -> pl.DataFrame:
                payloads = [data] if isinstance(data, bytes) else data
                return pl.read_json(b"[" + b",".join(payloads) + b"]")

            return decoder

        return None
