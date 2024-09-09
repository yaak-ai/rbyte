import json
from collections.abc import Hashable, Mapping
from functools import cached_property
from mmap import ACCESS_READ, mmap
from os import PathLike
from pathlib import Path
from typing import Literal, cast, override

import more_itertools as mit
import polars as pl
import polars._typing as plt
import pyarrow as pa
from cachetools import cached
from google.protobuf.descriptor_pb2 import FileDescriptorProto, FileDescriptorSet
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.message import Message
from google.protobuf.message_factory import GetMessageClassesForFiles
from mcap.reader import SeekingReader
from mcap.records import Schema
from ptars import HandlerPool
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tqdm import tqdm
from xxhash import xxh3_64_intdigest as digest

from rbyte.config.base import HydraConfig
from rbyte.io.table.base import TableReaderBase
from rbyte.utils.dataframe import unnest_all

from .config import Config

logger = get_logger(__name__)


class McapProtobufTableReader(TableReaderBase, Hashable):
    FRAME_INDEX_COLUMN_NAME: Literal["frame_idx"] = "frame_idx"

    def __init__(self, **kwargs: object) -> None:
        self._config = Config.model_validate(kwargs)

    @override
    def read(self, path: PathLike[str]) -> Mapping[str, pl.DataFrame]:
        with (
            bound_contextvars(path=str(path)),
            Path(path).open("rb") as _f,
            mmap(fileno=_f.fileno(), length=0, access=ACCESS_READ) as f,
        ):
            reader = SeekingReader(f, validate_crcs=self._config.validate_crcs)  # pyright: ignore[reportArgumentType]
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

            schemas = {
                channel.topic: summary.schemas[channel.schema_id]
                for channel in summary.channels.values()
                if channel.topic in topics
            }

            message_counts = (
                {
                    channel.topic: stats.channel_message_counts[channel_id]
                    for channel_id, channel in summary.channels.items()
                    if channel.topic in topics
                }
                if (stats := summary.statistics) is not None
                else {}
            )

            messages = mit.bucket(
                reader.iter_messages(topics),
                key=lambda x: x[1].topic,
                validator=lambda k: k in topics,
            )

            dfs: Mapping[str, pl.DataFrame] = {}
            handler_pool = HandlerPool()

            for topic, schema in self.schemas.items():
                log_time, publish_time, data = mit.unzip(
                    (msg.log_time, msg.publish_time, msg.data)
                    for (*_, msg) in tqdm(
                        messages[topic],
                        total=message_counts[topic],
                        postfix={"topic": topic},
                    )
                )

                msg_type = self._get_message_type(schemas[topic])
                handler = handler_pool.get_for_message(msg_type.DESCRIPTOR)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

                record_batch = cast(
                    pa.RecordBatch,
                    handler.list_to_record_batch(list(data)),  # pyright: ignore[reportUnknownMemberType]
                )
                df = cast(pl.DataFrame, pl.from_arrow(record_batch))  # pyright: ignore[reportUnknownMemberType]

                dfs[topic] = (
                    (df.select(unnest_all(df.collect_schema())))
                    .hstack([
                        pl.Series("log_time", log_time, pl.UInt64),
                        pl.Series("publish_time", publish_time, pl.UInt64),
                    ])
                    .select(schema)
                    .cast({k: v for k, v in schema.items() if v is not None})
                )

        for topic, df in dfs.items():
            match schemas[topic]:
                case Schema(
                    encoding="protobuf",
                    name="foxglove.CompressedImage"
                    | "foxglove.CompressedVideo"
                    | "foxglove.RawImage",
                ):
                    dfs[topic] = df.with_row_index(self.FRAME_INDEX_COLUMN_NAME)

                case _:
                    pass

        return dfs

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)

    @cached(cache={}, key=lambda _, schema: schema.name)  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType, reportUnknownMemberType]
    def _get_message_type(self, schema: Schema) -> type[Message]:  # noqa: PLR6301
        # inspired by https://github.com/foxglove/mcap/blob/e591defaa95186cef27e37c49fa7e1f0c9f2e8a6/python/mcap-protobuf-support/mcap_protobuf/decoder.py#L29
        fds = FileDescriptorSet.FromString(schema.data)
        descriptors = {fd.name: fd for fd in fds.file}
        pool = DescriptorPool()

        def _add(fd: FileDescriptorProto) -> None:
            for dependency in fd.dependency:
                if dependency in descriptors:
                    _add(descriptors.pop(dependency))

            pool.Add(fd)  # pyright: ignore[reportUnknownMemberType]

        while descriptors:
            _add(descriptors.popitem()[1])

        messages = GetMessageClassesForFiles([fd.name for fd in fds.file], pool)

        return mit.one(
            msg_type for name, msg_type in messages.items() if name == schema.name
        )

    @cached_property
    def schemas(self) -> dict[str, dict[str, plt.PolarsDataType | None]]:
        return {
            topic: {
                path: leaf.instantiate() if isinstance(leaf, HydraConfig) else leaf
                for path, leaf in fields.items()
            }
            for topic, fields in self._config.fields.items()
        }
