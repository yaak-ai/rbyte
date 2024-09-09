import json
from collections.abc import Hashable, Mapping
from functools import cached_property
from mmap import ACCESS_READ, mmap
from operator import attrgetter
from os import PathLike
from pathlib import Path
from typing import Any, Literal, override

import polars as pl
import polars._typing as plt
from mcap.reader import SeekingReader
from mcap_ros2.decoder import DecoderFactory
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from xxhash import xxh3_64_intdigest as digest

from rbyte.config.base import HydraConfig
from rbyte.io.table.base import TableReaderBase

from .config import Config

logger = get_logger(__name__)


class McapRos2TableReader(TableReaderBase, Hashable):
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
            reader = SeekingReader(
                f,  # pyright: ignore[reportArgumentType]
                validate_crcs=self._config.validate_crcs,
                decoder_factories=[DecoderFactory()],
            )
            summary = reader.get_summary()
            if summary is None:
                logger.error(msg := "missing summary")
                raise ValueError(msg)

            topics = self._config.fields.keys()
            if missing_topics := topics - (
                available_topics := {ch.topic for ch in summary.channels.values()}
            ):
                with bound_contextvars(
                    missing_topics=sorted(missing_topics),
                    available_topics=sorted(available_topics),
                ):
                    logger.error(msg := "missing topics")
                    raise ValueError(msg)

            getters = {
                topic: [attrgetter(field) for field in fields]
                for topic, fields in self.schemas.items()
            }

            rows: Mapping[str, list[list[Any]]] = {topic: [] for topic in self.schemas}

            for _, channel, msg, msg_decoded in reader.iter_decoded_messages(topics):
                topic = channel.topic
                row: list[Any] = []
                for getter in getters[topic]:
                    try:
                        attr = getter(msg_decoded)
                    except AttributeError:
                        attr = getter(msg)

                    row.append(attr)

                rows[topic].append(row)

        return {
            topic: pl.DataFrame(
                data=rows[topic],
                schema=self.schemas[topic],  # pyright: ignore[reportArgumentType]
                orient="row",
            )
            for topic in topics
        }

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)

    @cached_property
    def schemas(self) -> dict[str, dict[str, plt.PolarsDataType | None]]:
        return {
            topic: {
                path: leaf.instantiate() if isinstance(leaf, HydraConfig) else leaf
                for path, leaf in fields.items()
            }
            for topic, fields in self._config.fields.items()
        }
