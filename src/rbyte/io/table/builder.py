import operator
from collections import Counter
from collections.abc import Hashable, Sequence
from functools import reduce
from mmap import ACCESS_READ, mmap
from pathlib import Path
from typing import Annotated, Any, override

import more_itertools as mit
import polars as pl
from pydantic import ConfigDict, Field, FilePath, StringConstraints, validate_call
from structlog import get_logger
from xxhash import xxh3_64_intdigest as digest

from rbyte.config.base import BaseModel
from rbyte.io.table.base import (
    TableBuilderBase,
    TableCacheBase,
    TableMergerBase,
    TableReaderBase,
)

logger = get_logger(__name__)


class TableReaderConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: FilePath
    reader: TableReaderBase


class TableBuilder(TableBuilderBase):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        readers: Annotated[Sequence[TableReaderConfig], Field(min_length=1)],
        merger: TableMergerBase,
        filter: Annotated[str, StringConstraints(strip_whitespace=True)] | None = None,  # noqa: A002
        cache: TableCacheBase | None = None,
    ) -> None:
        super().__init__()

        self._readers = readers
        self._merger = merger
        self._filter = filter
        self._cache = cache

    def _build_cache_key(self) -> Hashable:
        from rbyte import __version__ as rbyte_version  # noqa: PLC0415

        key: list[Any] = [rbyte_version, hash(self._merger)]

        if self._filter is not None:
            key.append(digest(self._filter))

        for reader_config in self._readers:
            with (
                Path(reader_config.path).open("rb") as _f,
                mmap(_f.fileno(), 0, access=ACCESS_READ) as f,
            ):
                file_hash = digest(f)  # pyright: ignore[reportArgumentType]

            key.append((file_hash, hash(reader_config.reader)))

        return tuple(key)

    @override
    def build(self) -> pl.DataFrame:
        match self._cache:
            case TableCacheBase():
                key = self._build_cache_key()
                if key in self._cache:
                    logger.debug("reading table from cache")
                    df = self._cache.get(key)
                    if df is None:
                        raise RuntimeError

                    return df

                df = self._build()
                if not self._cache.set(key, df):
                    logger.warning("failed to cache table")

                return df

            case None:
                return self._build()

    def _build(self) -> pl.DataFrame:
        reader_dfs = [cfg.reader.read(cfg.path) for cfg in self._readers]
        if duplicate_keys := {
            k for k, count in Counter(mit.flatten(reader_dfs)).items() if count > 1
        }:
            logger.error(msg := "readers produced duplicate keys", keys=duplicate_keys)

            raise RuntimeError(msg)

        dfs = reduce(operator.or_, reader_dfs)
        df = self._merger.merge(dfs)

        return df.sql(f"select * from self where ({self._filter or True})")  # noqa: S608
