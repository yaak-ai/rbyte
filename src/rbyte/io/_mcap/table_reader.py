import json
from collections import defaultdict
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
from mcap.reader import SeekingReader
from optree import PyTree, tree_map
from polars._typing import PolarsDataType
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from pydantic import (
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
from rbyte.io.table.base import TableReader
from rbyte.utils.dataframe import unnest_all

logger = get_logger(__name__)


class Config(BaseModel):
    decoder_factories: frozenset[ImportString[type[DecoderFactory]]]
    fields: Mapping[str, Mapping[str, HydraConfig[PolarsDataType] | None]]
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
class SpecialField(StrEnum):
    log_time = "log_time"
    publish_time = "publish_time"
    idx = "_idx_"


class McapTableReader(TableReader, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config: Config = Config.model_validate(kwargs)

    @override
    def read(self, path: PathLike[str]) -> PyTree[pl.DataFrame]:
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

            topics = self._fields.keys()
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

            rows: Mapping[str, list[pl.DataFrame]] = defaultdict(list)

            for dmt in tqdm(
                reader.iter_decoded_messages(topics),
                desc="messages",
                total=message_count,
            ):
                schema = self._fields[dmt.channel.topic]
                message_fields, special_fields = map(
                    dict,
                    mit.partition(lambda kv: kv[0] in SpecialField, schema.items()),
                )

                special_fields = {
                    k: v for k, v in special_fields.items() if k != SpecialField.idx
                }

                row_df = pl.DataFrame(
                    [getattr(dmt.message, field) for field in special_fields],
                    schema=special_fields,
                )

                if (
                    message_df := self._build_message_df(
                        dmt.decoded_message, message_fields
                    )
                ) is not None:
                    row_df = row_df.hstack(message_df)

                rows[dmt.channel.topic].append(row_df)

        dfs: Mapping[str, pl.DataFrame] = {}
        for topic, row_dfs in rows.items():
            df = pl.concat(row_dfs, how="vertical")
            if (idx_name := SpecialField.idx) in (schema := self._fields[topic]):
                df = df.with_row_index(idx_name).cast({
                    idx_name: schema[idx_name] or pl.UInt32
                })

            dfs[topic] = df.rechunk()

        return dfs  # pyright: ignore[reportReturnType]

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)

    @cached_property
    def _fields(self) -> Mapping[str, Mapping[str, PolarsDataType | None]]:
        return tree_map(HydraConfig.instantiate, self._config.fields)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType, reportReturnType]

    @staticmethod
    def _build_message_df(
        message: object, fields: Mapping[str, PolarsDataType | None]
    ) -> pl.DataFrame | None:
        if not fields:
            return None

        df_schema = {name: dtype for name, dtype in fields.items() if dtype is not None}

        match message:
            case pl.DataFrame():
                return (
                    message.lazy()
                    .select(unnest_all(message.collect_schema()))
                    .select(fields)
                    .cast(df_schema)  # pyright: ignore[reportArgumentType]
                ).collect()

            case _:
                return pl.from_dict({
                    field: attrgetter(field)(message) for field in fields
                }).cast(df_schema)  # pyright: ignore[reportArgumentType]
