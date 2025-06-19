from os import PathLike
from typing import final

import duckdb
import polars as pl
from pydantic import validate_call

from .udf import functions_to_register


@final
class DuckDbDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __call__(self, *, query: str, path: PathLike[str]) -> pl.DataFrame:

        for func in functions_to_register:
            _ = duckdb.create_function(**func)

        return duckdb.sql(query=query.format(path=path)).pl()
