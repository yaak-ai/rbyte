from mmap import ACCESS_READ, mmap
from operator import itemgetter
from os import PathLike
from pathlib import Path
from typing import cast, final

import more_itertools as mit
import polars as pl
from google.protobuf.message import Message
from polars.datatypes import DataType
from ptars import HandlerPool
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tqdm import tqdm
from xxhash import xxh3_64_hexdigest as digest

from rbyte.config.base import PickleableImportString
from rbyte.io.yaak.proto import sensor_pb2

from .message_iterator import YaakMetadataMessageIterator

logger = get_logger(__name__)


type Fields = dict[
    PickleableImportString[type[Message]], dict[str, InstanceOf[DataType] | None]
]


@final
class YaakMetadataDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(self, *, fields: Fields) -> None:
        super().__init__()

        self._fields = fields

    def __pipefunc_hash__(self) -> str:  # noqa: PLW3201
        return digest(str(self._fields))

    def __call__(self, path: PathLike[str]) -> dict[str, pl.DataFrame]:
        with bound_contextvars(path=path):
            result = self._build(path)
            logger.debug(
                "built dataframes", length={k: len(v) for k, v in result.items()}
            )

            return result

    def _build(self, path: PathLike[str]) -> dict[str, pl.DataFrame]:
        with Path(path).open("rb") as _f, mmap(_f.fileno(), 0, access=ACCESS_READ) as f:
            handler_pool = HandlerPool()

            message_types = {k.obj for k in self._fields}
            messages = mit.bucket(
                YaakMetadataMessageIterator(f, message_types=message_types),
                key=itemgetter(0),
                validator=message_types.__contains__,
            )

            dfs = {
                msg.obj.__name__: cast(
                    pl.DataFrame,
                    pl.from_arrow(  # pyright: ignore[reportUnknownMemberType]
                        data=handler_pool.get_for_message(msg.obj.DESCRIPTOR)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                        .list_to_record_batch([
                            msg_data
                            for (_, msg_data) in tqdm(
                                messages[msg.obj], postfix={"msg": msg.obj}
                            )
                        ])
                        .select(schema),
                        schema=schema,
                        rechunk=True,
                    ),
                )
                for msg, schema in self._fields.items()
            }

        if (df := dfs.pop((k := sensor_pb2.ImageMetadata.__name__), None)) is not None:
            dfs |= {
                ".".join((k, *map(str, k_partition))): df_partition
                for k_partition, df_partition in df.partition_by(
                    "camera_name", include_key=False, as_dict=True
                ).items()
            }

        return dfs
