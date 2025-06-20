from dataclasses import asdict
from os import PathLike
from typing import final

import duckdb
import polars as pl
from pydantic import validate_call

from .udf import udf_list


@final
class DuckDbDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __call__(self, *, query: str, path: PathLike[str]) -> pl.DataFrame:
        for udf in udf_list:
            _ = duckdb.create_function(**asdict(udf))

        return duckdb.sql(query=query.format(path=path)).pl()
