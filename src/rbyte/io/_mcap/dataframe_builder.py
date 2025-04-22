from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum, unique
from mmap import ACCESS_READ, mmap
from operator import attrgetter
from os import PathLike
from pathlib import Path
from typing import NamedTuple, final

import more_itertools as mit
import polars as pl
from mcap.decoder import DecoderFactory
from mcap.reader import SeekingReader
from polars.datatypes import DataType
from pydantic import ImportString, InstanceOf, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tqdm import tqdm

from rbyte.utils.dataframe import unnest_all

logger = get_logger(__name__)


type Fields = dict[str, dict[str, InstanceOf[DataType] | None]]

type DecoderFactories = Sequence[
    ImportString[type[DecoderFactory]] | type[DecoderFactory]
]


class RowValues(NamedTuple):
    topic: str
    values: Iterable[object]


@unique
class SpecialField(StrEnum):
    log_time = "log_time"
    publish_time = "publish_time"


@final
class McapDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(
        self,
        *,
        decoder_factories: DecoderFactories,
        fields: Fields,
        validate_crcs: bool = True,
    ) -> None:
        self._decoder_factories = decoder_factories
        self._fields = fields
        self._validate_crcs = validate_crcs

    def __call__(self, path: PathLike[str]) -> dict[str, pl.DataFrame]:
        with bound_contextvars(path=path):
            result = self._build(path)
            logger.debug(
                "built dataframes", length={k: len(v) for k, v in result.items()}
            )

            return result

    def _build(self, path: PathLike[str]) -> dict[str, pl.DataFrame]:
        with (
            bound_contextvars(path=str(path)),
            Path(path).open("rb") as _f,
            mmap(fileno=_f.fileno(), length=0, access=ACCESS_READ) as f,
        ):
            reader = SeekingReader(
                f,  # pyright: ignore[reportArgumentType]
                validate_crcs=self._validate_crcs,
                decoder_factories=[f() for f in self._decoder_factories],
            )
            summary = reader.get_summary()
            if summary is None:
                logger.error(msg := "missing summary")
                raise ValueError(msg)

            topics_requested = self._fields.keys()
            topics_available = {channel.topic for channel in summary.channels.values()}
            if topics_missing := (topics_requested - topics_available):
                logger.warning("missing topics", topics=topics_missing)

            topics = topics_requested & topics_available
            message_count = (
                sum(
                    stats.channel_message_counts[channel.id]
                    for channel in summary.channels.values()
                    if channel.topic in topics
                )
                if (stats := summary.statistics) is not None
                else None
            )

            rows: dict[str, list[pl.DataFrame]] = defaultdict(list)

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

        return {
            topic: pl.concat(row_dfs, how="vertical", rechunk=True)
            for topic, row_dfs in rows.items()
        }

    @staticmethod
    def _build_message_df(
        message: object, fields: dict[str, DataType | None]
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
