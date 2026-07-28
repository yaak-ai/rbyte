from collections import defaultdict
from collections.abc import Callable, Sequence
from enum import StrEnum, unique
from functools import cached_property
from mmap import ACCESS_READ, mmap
from operator import attrgetter
from os import PathLike
from pathlib import Path
from typing import Any, final

import more_itertools as mit
import polars as pl
import polars.selectors as cs
from mcap.decoder import DecoderFactory
from mcap.exceptions import DecoderNotFoundError
from mcap.reader import SeekingReader
from polars.datatypes import DataType
from pydantic import ImportString, InstanceOf, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tqdm import tqdm

logger = get_logger(__name__)


type Fields = dict[str, dict[str, InstanceOf[DataType] | None]]


@unique
class SpecialField(StrEnum):
    log_time = "log_time"
    publish_time = "publish_time"


@final
class McapReader:
    @validate_call
    def __init__(
        self,
        *,
        decoder_factories: Sequence[ImportString[type[DecoderFactory]]]
        | Sequence[type[DecoderFactory]],
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
            Path(path).open("rb") as f_,
            mmap(fileno=f_.fileno(), length=0, access=ACCESS_READ) as f,
        ):
            reader = SeekingReader(
                f,  # ty: ignore[invalid-argument-type]
                validate_crcs=self._validate_crcs,
                decoder_factories=self._decoder_factories_instantiated,
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
            fields = {
                topic: tuple(
                    map(
                        dict,
                        mit.partition(
                            lambda kv: kv[0] in SpecialField,
                            self._fields[topic].items(),
                        ),
                    )
                )
                for topic in topics
            }
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
            decoders: dict[int, Callable[[bytes], Any]] = {}
            for schema, channel, message in tqdm(
                reader.iter_messages(topics, log_time_order=True),
                desc="messages",
                total=message_count,
            ):
                message_fields, special_fields = fields[channel.topic]

                row_df = pl.DataFrame(
                    {field: [getattr(message, field)] for field in special_fields},
                    schema=special_fields,
                )

                if message_fields:
                    try:
                        decoder = decoders[channel.id]
                    except KeyError:
                        decoder = mit.first_true(
                            (
                                factory.decoder_for(channel.message_encoding, schema)
                                for factory in self._decoder_factories_instantiated
                            ),
                            pred=lambda decoder: decoder is not None,
                        )
                        if decoder is None:
                            msg = (
                                "no decoder factory supplied for message encoding "
                                f"{channel.message_encoding}, schema {schema}"
                            )
                            raise DecoderNotFoundError(msg) from None
                        decoders[channel.id] = decoder

                    row_df = self._build_message_df(
                        decoder(message.data), message_fields
                    ).hstack(row_df)

                rows[channel.topic].append(row_df)

        return {
            topic: pl.concat(row_dfs, how="vertical", rechunk=True)
            for topic, row_dfs in rows.items()
        }

    @staticmethod
    def _build_message_df(
        message: object, fields: dict[str, DataType | None]
    ) -> pl.DataFrame:
        df_schema = {name: dtype for name, dtype in fields.items() if dtype is not None}

        match message:
            case pl.DataFrame():
                return (
                    message
                    .lazy()
                    .unnest(cs.struct(), separator=".")
                    .select(fields.keys())
                    .cast(df_schema)  # ty:ignore[invalid-argument-type]
                ).collect()

            case _:
                return pl.from_dict({
                    field: attrgetter(field)(message) for field in fields
                }).cast(df_schema)  # ty:ignore[invalid-argument-type]

    @cached_property
    def _decoder_factories_instantiated(self) -> tuple[DecoderFactory, ...]:
        return tuple(f() for f in self._decoder_factories)
