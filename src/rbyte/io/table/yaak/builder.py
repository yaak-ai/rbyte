import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from enum import StrEnum, auto
from functools import cached_property
from hashlib import blake2b
from mmap import ACCESS_READ, mmap
from operator import attrgetter
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar, cast, override

import polars as pl
from google._upb._message import (
    RepeatedScalarContainer,  # noqa: PLC2701 # pyright: ignore[reportUnknownVariableType]
)
from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp
from more_itertools import (
    all_unique,
    always_iterable,  # pyright: ignore[reportUnknownVariableType]
    collapse,  # pyright: ignore[reportUnknownVariableType]
    map_reduce,
)
from polars.type_aliases import AsofJoinStrategy
from pydantic import (
    Field,
    ImportString,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)
from structlog import get_logger
from structlog.contextvars import bound_contextvars

from rbyte.config.base import BaseModel, HydraConfig
from rbyte.io.table.base import TableBuilder
from rbyte.utils.dataframe_cache import DataframeDiskCache

from .message_iterator import YaakMetadataMessageIterator
from .proto import can_pb2, sensor_pb2

logger = get_logger(__name__)

K = TypeVar("K")
V = TypeVar("V")


def flatten_mapping(mapping: Mapping[K, V]) -> Iterator[tuple[tuple[K, ...], V]]:
    for k, v in mapping.items():
        if isinstance(v, Mapping):
            yield from (
                ((k, *always_iterable(_k)), _v)  # pyright: ignore[reportUnknownArgumentType]
                for _k, _v in flatten_mapping(v)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            )
        else:
            yield (k,), v


class CameraName(StrEnum):
    CAM_FRONT_CENTER = auto()
    CAM_FRONT_LEFT = auto()
    CAM_FRONT_RIGHT = auto()
    CAM_LEFT_FORWARD = auto()
    CAM_LEFT_BACKWARD = auto()
    CAM_RIGHT_FORWARD = auto()
    CAM_RIGHT_BACKWARD = auto()
    CAM_REAR = auto()


class AsofMergeConfig(BaseModel):
    method: Literal["asof"] = "asof"
    tolerance: Annotated[
        str, StringConstraints(strip_whitespace=True, to_lower=True, pattern=r"\d+ms$")
    ] = "100ms"
    strategy: AsofJoinStrategy = "nearest"


class InterpMergeConfig(BaseModel):
    method: Literal["interp"] = "interp"


MergeConfig = AsofMergeConfig | InterpMergeConfig


class MetadataFieldSelectConfig(BaseModel):
    name: str
    merge: MergeConfig | None = None
    partition: bool = False


class MetadataSelectMessageConfig(BaseModel):
    type: ImportString[type[Message]]
    alias: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class MetadataSelectConfig(BaseModel):
    message: MetadataSelectMessageConfig
    fields: tuple[MetadataFieldSelectConfig, ...] = Field(min_length=1)

    @field_validator("fields", mode="before")
    @classmethod
    def validate_fields(
        cls, fields: tuple[MetadataFieldSelectConfig, ...]
    ) -> tuple[MetadataFieldSelectConfig, ...]:
        if not all_unique(fields, key=attrgetter("name")):
            msg = "field names not unique"
            raise ValueError(msg)

        return fields

    @model_validator(mode="after")
    def validate_message_fields(self) -> Self:
        valid = set(self.message.type.DESCRIPTOR.fields_by_name.keys())
        if invalid := {f.name for f in self.fields} - valid:
            msg = f"invalid fields for `{self.message.type}`: {invalid}"
            raise ValueError(msg)

        return self

    @field_serializer("fields")
    @staticmethod
    def serialize_fields(
        fields: Iterable[MetadataFieldSelectConfig],
    ) -> tuple[MetadataFieldSelectConfig, ...]:
        return tuple(sorted(fields, key=attrgetter("name")))


class MetadataMergeReferenceConfig(BaseModel):
    key: tuple[ImportString[type[Message]]] | tuple[ImportString[type[Message]], str]
    column: str


class MetadataMergeConfig(BaseModel):
    reference: MetadataMergeReferenceConfig


class MetadataTableBuilderConfig(BaseModel):
    cameras: frozenset[CameraName] = Field(default=frozenset(CameraName))
    select: tuple[MetadataSelectConfig, ...] = Field(min_length=1)
    merge: MetadataMergeConfig
    filter: Annotated[str, StringConstraints(strip_whitespace=True)] | None = None
    cache: HydraConfig[DataframeDiskCache] | None = None

    @field_validator("select")
    @classmethod
    def validate_select(
        cls, select: tuple[MetadataSelectConfig, ...]
    ) -> tuple[MetadataSelectConfig, ...]:
        if not all_unique(select, key=attrgetter("message.type")):
            msg = "select message paths not unique"
            raise ValueError(msg)

        if not all_unique(select, key=attrgetter("message.alias")):
            msg = "select message aliases not unique"
            raise ValueError(msg)

        return select

    @field_serializer("cameras")
    @staticmethod
    def serialize_cameras(cameras: Iterable[CameraName]) -> tuple[CameraName, ...]:
        return tuple(sorted(cameras))

    @field_serializer("select")
    @staticmethod
    def serialize_select(
        select: Iterable[MetadataSelectConfig],
    ) -> tuple[MetadataSelectConfig, ...]:
        return tuple(sorted(select, key=attrgetter("message.alias")))


class YaakMetadataTableBuilder(TableBuilder):
    COLUMN_NAME_SEPARATOR = "."
    SCHEMA_OVERRIDES: Mapping[type[Message], pl._typing.SchemaDict] = {
        sensor_pb2.ImageMetadata: {
            "time_stamp": pl.Datetime("us"),
            "camera_name": pl.Enum(tuple(CameraName)),
        },
        sensor_pb2.Gnss: {"time_stamp": pl.Datetime("us")},
        can_pb2.VehicleMotion: {"time_stamp": pl.Datetime("us")},
        can_pb2.VehicleState: {"time_stamp": pl.Datetime("us")},
    }

    def __init__(self, config: object) -> None:
        super().__init__()

        self.config = MetadataTableBuilderConfig.model_validate(config)

    @cached_property
    def _dataframe_cache(self) -> DataframeDiskCache | None:
        return cfg.instantiate() if (cfg := self.config.cache) is not None else None

    def _build_dataframe_cache_key(self, path: PathLike[str]) -> tuple[str, str]:
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_json = self.config.model_dump_json(exclude={"cache"})
        config_dict = json.loads(config_json)
        config_json_sorted = json.dumps(config_dict, sort_keys=True)
        config_hash = blake2b(config_json_sorted.encode("utf-8")).hexdigest()

        with (
            Path(path).open("rb") as f,
            mmap(f.fileno(), 0, access=ACCESS_READ) as file,
        ):
            file_hash = blake2b(file).hexdigest()

        return (config_hash, file_hash)

    @override
    def build(self, path: PathLike[str]) -> pl.DataFrame:
        with bound_contextvars(path=str(path)):
            match self._dataframe_cache:
                case None:
                    return self._build_dataframe(path)

                case _:
                    key = self._build_dataframe_cache_key(path)
                    match df := self._dataframe_cache.get(key):
                        case None:
                            logger.debug("dataframe cache miss")
                            df = self._build_dataframe(path)
                            if not self._dataframe_cache.set(key, df):
                                logger.warning("failed to cache dataframe")

                            return df

                        case _:
                            logger.debug("dataframe cache hit")
                            return df

    def _build_dataframe(self, path: PathLike[str]) -> pl.DataFrame:
        logger.debug("building dataframe")

        dfs = self._read_message_dataframes(path)
        dfs = self._partition_message_dataframes(dfs)
        df = self._merge_message_dataframes(dfs)

        return df.sql(
            f"select * from self where ({self.config.filter or True})"  # noqa: S608
        )

    def _read_message_dataframes(
        self, path: PathLike[str]
    ) -> Mapping[type[Message], pl.DataFrame]:
        message_rows: Mapping[type[Message], list[list[Message]]] = {
            m: [] for m in self._select_fields
        }

        with Path(path).open("rb") as _f, mmap(_f.fileno(), 0, access=ACCESS_READ) as f:
            messages = YaakMetadataMessageIterator(f, message_types=message_rows.keys())
            for message in messages:
                # PERF: avoid including messages from cameras we don't want
                if (
                    isinstance(message, sensor_pb2.ImageMetadata)
                    and message.camera_name not in self.config.cameras
                ):
                    continue

                msg_type = type(message)

                row: list[Any] = []
                for field in self._select_fields[msg_type]:
                    match attr := getattr(message, field):
                        case Timestamp():
                            row.append(attr.ToMicroseconds())

                        case RepeatedScalarContainer():
                            row.append(tuple(attr))  # pyright: ignore[reportUnknownArgumentType]

                        case _:
                            row.append(attr)

                message_rows[msg_type].append(row)

        message_dfs: Mapping[type[Message], pl.DataFrame] = {
            msg_type: pl.DataFrame(
                data=rows,
                schema=self._select_fields[msg_type],
                schema_overrides=self.SCHEMA_OVERRIDES.get(msg_type, None),
                orient="row",
            )
            for msg_type, rows in message_rows.items()
        }

        return message_dfs

    def _partition_message_dataframes(
        self, message_dfs: Mapping[type[Message], pl.DataFrame]
    ) -> Mapping[
        type[Message], pl.DataFrame | Mapping[tuple[object, ...], pl.DataFrame]
    ]:
        return {
            msg_type: df
            if (by := self._partition_fields.get(msg_type, None)) is None
            else df.partition_by(*by, include_key=False, as_dict=True)
            for msg_type, df in message_dfs.items()
        }

    def _merge_message_dataframes(
        self,
        message_dfs: Mapping[
            type[Message], pl.DataFrame | Mapping[tuple[object, ...], pl.DataFrame]
        ],
    ) -> pl.DataFrame:
        ref_df_key, ref_col = (
            self.config.merge.reference.key,
            self.config.merge.reference.column,
        )

        dfs = {
            (merge_df_key := tuple(collapse(_k))): cast(pl.DataFrame, v)
            .sort(ref_col)
            .rename(lambda col: self._col_name(merge_df_key, col))
            for _k, v in flatten_mapping(message_dfs)
        }

        try:
            ref_df = dfs.pop(ref_df_key)
        except KeyError as e:
            msg = f"invalid reference key {ref_df_key}, valid: {list(dfs)}"
            raise ValueError(msg) from e

        ref_df_ref_col = self._col_name(*ref_df_key, ref_col)

        for merge_df_key, merge_df in dfs.items():
            msg_type, *_ = merge_df_key
            merge_df_ref_col = self._col_name(merge_df_key, ref_col)

            for merge_config, merge_fields in self._merge_fields[msg_type].items():
                merge_df_cols = tuple(
                    self._col_name(merge_df_key, field) for field in merge_fields
                )

                match merge_config:
                    case AsofMergeConfig(strategy=strategy, tolerance=tolerance):
                        ref_df = ref_df.join_asof(
                            merge_df.select(merge_df_ref_col, *merge_df_cols),
                            left_on=ref_df_ref_col,
                            right_on=merge_df_ref_col,
                            strategy=strategy,
                            tolerance=tolerance,
                        ).drop_nulls()

                    case InterpMergeConfig():
                        ref_df = (
                            # take a union of timestamps
                            ref_df.join(
                                merge_df.select(merge_df_ref_col, *merge_df_cols),
                                how="full",
                                left_on=ref_df_ref_col,
                                right_on=merge_df_ref_col,
                                coalesce=True,
                            )
                            # interpolate
                            .with_columns(
                                pl.col(merge_df_cols).interpolate_by(ref_df_ref_col)
                            )
                            # narrow back to original ref col
                            .join(
                                ref_df.select(ref_df_ref_col),
                                on=ref_df_ref_col,
                                how="semi",
                            )
                            .sort(ref_df_ref_col)
                        )

        return ref_df

    @cached_property
    def _message_type_aliases(self) -> Mapping[type[Message], str]:
        return {
            select.message.type: select.message.alias for select in self.config.select
        }

    @cached_property
    def _select_fields(self) -> Mapping[type[Message], Sequence[str]]:
        return {
            select.message.type: tuple(field.name for field in select.fields)
            for select in self.config.select
        }

    @cached_property
    def _merge_fields(
        self,
    ) -> Mapping[type[Message], Mapping[MergeConfig, Sequence[str]]]:
        return {
            select.message.type: map_reduce(
                (field for field in select.fields if field.merge is not None),
                keyfunc=attrgetter("merge"),
                valuefunc=attrgetter("name"),
            )
            for select in self.config.select
        }

    @cached_property
    def _partition_fields(self) -> Mapping[type[Message], Sequence[str]]:
        return {
            select.message.type: field_names
            for select in self.config.select
            if (
                field_names := tuple(
                    field.name for field in select.fields if field.partition
                )
            )
        }

    def _col_name(self, *args: object) -> str:
        return self.COLUMN_NAME_SEPARATOR.join(
            self._message_type_aliases.get(arg, arg) for arg in collapse(args)
        )
