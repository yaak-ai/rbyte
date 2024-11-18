from collections.abc import Hashable, Mapping
from mmap import ACCESS_READ, mmap
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, override

import polars as pl
from optree import PyTree, tree_map
from pydantic import Field, StringConstraints, validate_call
from structlog import get_logger
from xxhash import xxh3_64_intdigest as digest

from rbyte.config import BaseModel

from .base import TableBuilder as _TableBuilder
from .base import TableCache, TableMerger, TableReader

logger = get_logger(__name__)


class TableReaderConfig(BaseModel):
    path: PathLike[str]
    reader: TableReader


class TableBuilder(_TableBuilder):
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self,
        readers: Annotated[Mapping[str, TableReaderConfig], Field(min_length=1)],
        merger: TableMerger,
        filter: Annotated[str, StringConstraints(strip_whitespace=True)] | None = None,  # noqa: A002
        cache: TableCache | None = None,
    ) -> None:
        super().__init__()

        self._readers: Mapping[str, TableReaderConfig] = readers
        self._merger: TableMerger = merger
        self._filter: str | None = filter
        self._cache: TableCache | None = cache

    def _build_cache_key(self) -> Hashable:
        from rbyte import __version__  # noqa: PLC0415

        key: list[Any] = [__version__, hash(self._merger)]

        if self._filter is not None:
            key.append(digest(self._filter))

        for reader_name, reader_config in sorted(self._readers.items()):
            with (
                Path(reader_config.path).open("rb") as _f,
                mmap(_f.fileno(), 0, access=ACCESS_READ) as f,
            ):
                file_hash = digest(f)  # pyright: ignore[reportArgumentType]

            key.append((file_hash, digest(reader_name), hash(reader_config.reader)))

        return tuple(key)

    @override
    def build(self) -> pl.DataFrame:
        match self._cache:
            case TableCache():
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
        dfs: PyTree[pl.DataFrame] = tree_map(
            lambda cfg: cfg.reader.read(cfg.path),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownLambdaType]
            self._readers,  # pyright: ignore[reportArgumentType]
        )
        df = self._merger.merge(dfs)

        return df.sql(f"select * from self where ({self._filter or True})")  # noqa: S608
