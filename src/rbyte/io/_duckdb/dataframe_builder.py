from os import PathLike
from typing import final

import duckdb
import polars as pl
from pydantic import validate_call


@final
class DuckDbDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __call__(self, *, query: str, path: PathLike[str]) -> pl.DataFrame:
        return duckdb.sql(query=query.format(path=path)).pl()
