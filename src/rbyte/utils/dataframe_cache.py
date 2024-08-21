from collections.abc import Hashable
from io import BufferedReader
from tempfile import TemporaryFile
from typing import Literal

import polars as pl
from diskcache import Cache
from pydantic import ByteSize, DirectoryPath, NewPath, validate_call


class DataframeDiskCache:
    @validate_call
    def __init__(
        self, directory: DirectoryPath | NewPath, size_limit: ByteSize | None = None
    ) -> None:
        super().__init__()
        self._cache = Cache(directory=directory, size_limit=size_limit)

    def get(self, key: Hashable) -> pl.DataFrame | None:
        match val := self._cache.get(key, default=None, read=True):  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            case BufferedReader():
                return pl.read_ipc(val)

            case None:
                return None

            case _:  # pyright: ignore[reportUnknownVariableType]
                raise NotImplementedError

    def set(self, key: Hashable, dataframe: pl.DataFrame) -> Literal[True]:
        with TemporaryFile() as f:
            dataframe.write_ipc(f, compression="uncompressed")
            f.seek(0)  # pyright: ignore[reportUnusedCallResult]
            return self._cache.set(key, f, read=True)  # pyright: ignore[reportUnknownMemberType]
