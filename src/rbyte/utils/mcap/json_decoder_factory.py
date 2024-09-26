import json
from collections.abc import Callable
from typing import override

from box import Box
from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from structlog import get_logger

logger = get_logger(__name__)


class McapJsonDecoderFactory(McapDecoderFactory):
    @override
    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes], Box] | None:
        match message_encoding, getattr(schema, "encoding", None):
            case "json", "jsonschema":
                return self._decoder

            case _:
                return None

    @staticmethod
    def _decoder(data: bytes) -> Box:
        return Box(json.loads(data))
