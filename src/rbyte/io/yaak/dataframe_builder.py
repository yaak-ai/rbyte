from collections.abc import Mapping
from mmap import ACCESS_READ, mmap
from operator import itemgetter
from os import PathLike
from pathlib import Path
from typing import cast, final

import more_itertools as mit
import polars as pl
from google.protobuf.message import Message
from polars._typing import PolarsDataType  # noqa: PLC2701
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from ptars import HandlerPool
from pydantic import ConfigDict, ImportString, validate_call
from structlog import get_logger
from tqdm import tqdm

from .message_iterator import YaakMetadataMessageIterator
from .proto import sensor_pb2

logger = get_logger(__name__)


type Fields = Mapping[
    type[Message] | ImportString[type[Message]], Mapping[str, PolarsDataType | None]
]


@final
class YaakMetadataDataFrameBuilder:
    __name__ = __qualname__

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, *, fields: Fields) -> None:
        super().__init__()

        self._fields = fields

    def __call__(self, path: PathLike[str]) -> Mapping[str, pl.DataFrame]:
        with Path(path).open("rb") as _f, mmap(_f.fileno(), 0, access=ACCESS_READ) as f:
            handler_pool = HandlerPool()

            messages = mit.bucket(
                YaakMetadataMessageIterator(f, message_types=self._fields),
                key=itemgetter(0),
                validator=self._fields.__contains__,
            )

            dfs = {
                msg_type.__name__: cast(
                    pl.DataFrame,
                    pl.from_arrow(  # pyright: ignore[reportUnknownMemberType]
                        data=handler_pool.get_for_message(msg_type.DESCRIPTOR)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                        .list_to_record_batch([
                            msg_data
                            for (_, msg_data) in tqdm(
                                messages[msg_type], postfix={"msg_type": msg_type}
                            )
                        ])
                        .select(schema),
                        schema=schema,
                        rechunk=True,
                    ),
                )
                for msg_type, schema in self._fields.items()
            }

        if (df := dfs.pop((k := sensor_pb2.ImageMetadata.__name__), None)) is not None:
            dfs |= {
                ".".join((k, *map(str, k_partition))): df_partition
                for k_partition, df_partition in df.partition_by(
                    "camera_name", include_key=False, as_dict=True
                ).items()
            }

        return dfs


# exposing all kwargs so its cacheable by pipefunc
def build_yaak_metadata_dataframe(
    *, path: PathLike[str], fields: Fields
) -> Mapping[str, pl.DataFrame]:
    return YaakMetadataDataFrameBuilder(fields=fields)(path)
