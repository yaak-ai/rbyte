import json
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import StrEnum, unique
from functools import cached_property
from mmap import ACCESS_READ, mmap
from operator import attrgetter
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, override

import more_itertools as mit
import polars as pl
from mcap.decoder import DecoderFactory
from mcap.reader import DecodedMessageTuple, SeekingReader
from polars._typing import PolarsDataType
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from pydantic import (
    ConfigDict,
    ImportString,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_serializer,
)
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tqdm import tqdm
from xxhash import xxh3_64_intdigest as digest

from rbyte.config.base import BaseModel, HydraConfig
from rbyte.io.table.base import TableReaderBase

logger = get_logger(__name__)


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    decoder_factories: Sequence[ImportString[type[DecoderFactory]]]

    fields: Mapping[
        str,
        Mapping[str, HydraConfig[PolarsDataType] | ImportString[PolarsDataType] | None],
    ]

    validate_crcs: bool = False

    @field_serializer("decoder_factories", when_used="json", mode="wrap")
    @staticmethod
    def serialize_decoder_factories(
        value: frozenset[ImportString[type[DecoderFactory]]],
        nxt: SerializerFunctionWrapHandler,
        _info: SerializationInfo,
    ) -> Sequence[str]:
        return sorted(nxt(value))


class RowValues(NamedTuple):
    topic: str
    values: Iterable[Any]


@unique
class SpecialFields(StrEnum):
    log_time = "log_time"
    publish_time = "publish_time"
    idx = "_idx_"


class McapTableReader(TableReaderBase, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config = Config.model_validate(kwargs)

    @override
    def read(self, path: PathLike[str]) -> Mapping[str, pl.DataFrame]:
        with (
            bound_contextvars(path=str(path)),
            Path(path).open("rb") as _f,
            mmap(fileno=_f.fileno(), length=0, access=ACCESS_READ) as f,
        ):
            reader = SeekingReader(
                f,  # pyright: ignore[reportArgumentType]
                validate_crcs=self._config.validate_crcs,
                decoder_factories=[f() for f in self._config.decoder_factories],
            )
            summary = reader.get_summary()
            if summary is None:
                logger.error(msg := "missing summary")
                raise ValueError(msg)

            topics = self.schemas.keys()
            if missing_topics := topics - (
                available_topics := {ch.topic for ch in summary.channels.values()}
            ):
                with bound_contextvars(
                    missing_topics=sorted(missing_topics),
                    available_topics=sorted(available_topics),
                ):
                    logger.error(msg := "missing topics")
                    raise ValueError(msg)

            message_count = (
                sum(
                    stats.channel_message_counts[channel.id]
                    for channel in summary.channels.values()
                    if channel.topic in topics
                )
                if (stats := summary.statistics) is not None
                else None
            )

            row_values = (
                RowValues(
                    dmt.channel.topic,
                    self._get_values(dmt, self.schemas[dmt.channel.topic]),
                )
                for dmt in tqdm(
                    reader.iter_decoded_messages(topics),
                    desc="messages",
                    total=message_count,
                )
            )

            row_values_by_topic = mit.bucket(row_values, key=lambda rd: rd.topic)

            dfs: Mapping[str, pl.DataFrame] = {}
            for topic, schema in self.schemas.items():
                df_schema = {k: v for k, v in schema.items() if k != SpecialFields.idx}
                df = pl.DataFrame(
                    data=(tuple(x.values) for x in row_values_by_topic[topic]),
                    schema=df_schema,  # pyright: ignore[reportArgumentType]
                    orient="row",
                )

                if (idx_name := SpecialFields.idx) in schema:
                    df = df.with_row_index(idx_name).cast({
                        idx_name: schema[idx_name] or pl.UInt32
                    })

                dfs[topic] = df

        return dfs

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)

    @cached_property
    def schemas(self) -> Mapping[str, Mapping[str, PolarsDataType | None]]:
        return {
            topic: {
                path: leaf.instantiate() if isinstance(leaf, HydraConfig) else leaf
                for path, leaf in fields.items()
            }
            for topic, fields in self._config.fields.items()
        }

    @staticmethod
    def _get_values(dmt: DecodedMessageTuple, fields: Iterable[str]) -> Iterable[Any]:
        for field in fields:
            match field:
                case SpecialFields.log_time:
                    yield dmt.message.log_time

                case SpecialFields.publish_time:
                    yield dmt.message.publish_time

                case SpecialFields.idx:
                    pass  # added later

                case _:
                    yield attrgetter(field)(dmt.decoded_message)
