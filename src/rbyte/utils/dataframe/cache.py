from collections.abc import Hashable
from io import BufferedReader
from tempfile import TemporaryFile
from typing import Literal, override

import polars as pl
from diskcache import Cache
from pydantic import ByteSize, DirectoryPath, NewPath, validate_call

from rbyte.io.table.base import TableCache


class DataframeDiskCache(TableCache):
    @validate_call
    def __init__(
        self, directory: DirectoryPath | NewPath, size_limit: ByteSize | None = None
    ) -> None:
        super().__init__()
        self._cache: Cache = Cache(directory=directory, size_limit=size_limit)

    @override
    def __contains__(self, key: Hashable) -> bool:
        return key in self._cache

    @override
    def get(self, key: Hashable) -> pl.DataFrame | None:
        match val := self._cache.get(key, default=None, read=True):  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            case BufferedReader():
                return pl.read_ipc(val)

            case None:
                return None

            case _:  # pyright: ignore[reportUnknownVariableType]
                raise NotImplementedError

    @override
    def set(self, key: Hashable, value: pl.DataFrame) -> Literal[True]:
        with TemporaryFile() as f:
            value.write_ipc(f, compression="uncompressed")
            f.seek(0)  # pyright: ignore[reportUnusedCallResult]
            return self._cache.set(key, f, read=True)  # pyright: ignore[reportUnknownMemberType]
