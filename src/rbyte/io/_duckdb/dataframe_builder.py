from dataclasses import asdict
from os import PathLike
from typing import final

import duckdb
import polars as pl
from pydantic import validate_call
from structlog import get_logger

from .udf import udf_list

logger = get_logger(__name__)


@final
class DuckDbDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __call__(self, *, query: str, path: PathLike[str]) -> pl.DataFrame:
        for udf in udf_list:
            try:
                _ = duckdb.create_function(**asdict(udf))
            except duckdb.NotImplementedException:
                msg = f"UDF {udf.name} was already registered"
                logger.info(msg)

        return duckdb.sql(query=query.format(path=path)).pl()
