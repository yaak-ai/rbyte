from collections.abc import Callable
from typing import override

import polars as pl
from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from structlog import get_logger

logger = get_logger(__name__)


class JsonMcapDecoderFactory(McapDecoderFactory):
    @override
    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes], pl.DataFrame] | None:
        if (
            message_encoding == "json"
            and schema is not None
            and schema.encoding == "jsonschema"
        ):
            return pl.read_json

        return None
