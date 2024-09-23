from collections.abc import Hashable
from mmap import ACCESS_READ, mmap
from os import PathLike
from pathlib import Path
from typing import Annotated, override

import polars as pl
from pydantic import ConfigDict, StringConstraints, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from xxhash import xxh3_64_intdigest as digest

from rbyte.io.table.base import (
    TableBuilderBase,
    TableCacheBase,
    TableMergerBase,
    TableReaderBase,
)

logger = get_logger(__name__)


class TableBuilder(TableBuilderBase):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        reader: TableReaderBase,
        merger: TableMergerBase,
        filter: Annotated[str, StringConstraints(strip_whitespace=True)] | None = None,  # noqa: A002
        cache: TableCacheBase | None = None,
    ) -> None:
        super().__init__()

        self._reader = reader
        self._merger = merger
        self._filter = filter
        self._cache = cache

    def _build_cache_key(self, path: PathLike[str]) -> Hashable:
        from rbyte import __version__ as rbyte_version  # noqa: PLC0415

        key = [rbyte_version, hash(self._reader), hash(self._merger)]

        if self._filter is not None:
            key.append(digest(self._filter))

        with Path(path).open("rb") as _f, mmap(_f.fileno(), 0, access=ACCESS_READ) as f:
            key.append(digest(f))  # pyright: ignore[reportArgumentType]

        return tuple(key)

    @override
    def build(self, path: PathLike[str]) -> pl.DataFrame:
        with bound_contextvars(path=str(path)):
            match self._cache:
                case TableCacheBase():
                    key = self._build_cache_key(path)
                    if key in self._cache:
                        logger.debug("reading table from cache")
                        df = self._cache.get(key)
                        if df is None:
                            raise RuntimeError

                        return df

                    df = self._build(path)
                    if not self._cache.set(key, df):
                        logger.warning("failed to cache table")

                    return df

                case None:
                    return self._build(path)

    def _build(self, path: PathLike[str]) -> pl.DataFrame:
        dfs = self._reader.read(path)
        df = self._merger.merge(dfs)
        return df.sql(f"select * from self where ({self._filter or True})")  # noqa: S608
