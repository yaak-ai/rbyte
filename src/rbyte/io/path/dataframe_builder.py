import os
from collections.abc import Iterator
from functools import cached_property
from typing import Self, final

import polars as pl
from optree import tree_map
from polars.datatypes import DataType
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    InstanceOf,
    field_serializer,
    model_validator,
    validate_call,
)
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from xxhash import xxh3_64_hexdigest as digest

logger = get_logger(__name__)


type Fields = dict[str, InstanceOf[DataType] | None]


def scantree(path: str) -> Iterator[str]:
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry.path


class Config(BaseModel):
    fields: Fields
    pattern: str

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if set(self.fields) != set(
            pl.Series(dtype=pl.String).str.extract_groups(self.pattern).struct.fields
        ):
            logger.error(
                msg := "field keys don't match pattern groups",
                fields=self.fields,
                pattern=self.pattern,
            )

            raise ValueError(msg)

        return self

    @field_serializer("fields", when_used="json")
    @staticmethod
    def _serialize_fields(fields: Fields) -> dict[str, str | None]:
        return tree_map(DataType._string_repr, fields)  # ty: ignore[invalid-argument-type, invalid-return-type]  # noqa: SLF001


@final
class PathDataFrameBuilder:
    __name__ = __qualname__  # ty: ignore[unresolved-reference]

    def __init__(self, *, fields: Fields, pattern: str) -> None:
        self._config = Config(fields=fields, pattern=pattern)

    def __pipefunc_hash__(self) -> str:  # noqa: PLW3201
        return digest(self._config.model_dump_json())

    @validate_call
    def __call__(self, path: DirectoryPath) -> pl.DataFrame:
        path_str = path.resolve().as_posix()
        with bound_contextvars(path=path_str):
            result = self._build(path_str)
            logger.debug("built dataframe", length=len(result))

            return result

    def _build(self, path: str) -> pl.DataFrame:
        return (
            pl.LazyFrame({"path": scantree(path)})
            .select(
                pl.col("path")
                .str.strip_prefix(path)
                .str.strip_prefix("/")
                .str.extract_groups(self._config.pattern)
                .alias("groups")
            )
            .unnest("groups")
            .drop_nulls()
            .select(self._config.fields.keys())
            .cast(self._schema, strict=True)
            .collect()
        )

    @cached_property
    def _schema(self) -> dict[str, DataType]:
        return {
            name: dtype
            for name, dtype in self._config.fields.items()
            if dtype is not None
        }
