import os
from collections.abc import Iterator, Mapping
from functools import cached_property
from typing import final

import polars as pl
from polars._typing import PolarsDataType  # noqa: PLC2701
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from pydantic import ConfigDict, DirectoryPath, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars

logger = get_logger(__name__)


type Fields = Mapping[str, PolarsDataType | None]


def scantree(path: str) -> Iterator[str]:
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry.path


@final
class PathDataFrameBuilder:
    __name__ = __qualname__

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, *, fields: Fields, pattern: str) -> None:
        if set(fields) != set(
            pl.Series(dtype=pl.String).str.extract_groups(pattern).struct.fields
        ):
            logger.error(
                msg := "field keys don't match pattern groups",
                fields=fields,
                pattern=pattern,
            )

            raise ValueError(msg)

        self._fields = fields
        self._pattern = pattern

    @validate_call
    def __call__(self, path: DirectoryPath) -> pl.DataFrame:
        path_str = path.resolve().as_posix()
        with bound_contextvars(path=path_str):
            result = self._build(path_str)
            logger.debug("built dataframe", length=len(result))

            return result

    def _build(self, path: str) -> pl.DataFrame:
        return (
            pl.LazyFrame({"path": scantree(path)})  # pyright: ignore[reportArgumentType]
            .select(
                pl.col("path")
                .str.strip_prefix(path)
                .str.strip_prefix("/")
                .str.extract_groups(self._pattern)
                .alias("groups")
            )
            .unnest("groups")
            .drop_nulls()
            .select(self._fields)
            .cast(self._schema, strict=True)  # pyright: ignore[reportArgumentType]
            .collect()
        )

    @cached_property
    def _schema(self) -> Mapping[str, PolarsDataType]:
        return {
            name: dtype for name, dtype in self._fields.items() if dtype is not None
        }
