import json
from collections.abc import Mapping
from functools import cached_property
from mmap import ACCESS_READ, mmap
from operator import itemgetter
from os import PathLike
from pathlib import Path
from typing import cast, override

import more_itertools as mit
import polars as pl
from google.protobuf.message import Message
from optree import PyTree, tree_map
from polars._typing import PolarsDataType
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from ptars import HandlerPool
from pydantic import ImportString
from structlog import get_logger
from tqdm import tqdm
from xxhash import xxh3_64_intdigest as digest

from rbyte.config.base import BaseModel, HydraConfig
from rbyte.io.table.base import TableReader

from .message_iterator import YaakMetadataMessageIterator
from .proto import sensor_pb2

logger = get_logger(__name__)


class Config(BaseModel):
    fields: Mapping[
        ImportString[type[Message]], Mapping[str, HydraConfig[PolarsDataType] | None]
    ]


class YaakMetadataTableReader(TableReader):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()

        self._config: Config = Config.model_validate(kwargs)

    @override
    def read(self, path: PathLike[str]) -> PyTree[pl.DataFrame]:
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

        return dfs  # pyright: ignore[reportReturnType]

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)

    @cached_property
    def _fields(self) -> Mapping[type[Message], Mapping[str, PolarsDataType | None]]:
        return tree_map(HydraConfig.instantiate, self._config.fields)  # pyright: ignore[reportArgumentType, reportUnknownVariableType, reportReturnType, reportUnknownMemberType, reportUnknownArgumentType]
