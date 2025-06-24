from collections.abc import Callable, Sequence
from typing import Any, TypedDict

import duckdb
from duckdb.typing import DuckDBPyType
from pydantic import ImportString, InstanceOf


class DuckDbUdfKwargs(TypedDict):
    name: str
    function: Callable[..., Any]
    parameters: list[InstanceOf[DuckDBPyType]] | None
    return_type: InstanceOf[DuckDBPyType] | None


def register_duckdb_udf(
    udfs: Sequence[ImportString[type[DuckDbUdfKwargs]] | DuckDbUdfKwargs] | None,
) -> None:
    if udfs is None:
        return

    for udf in udfs:
        try:
            duckdb.create_function(**udf)  # pyright: ignore[reportCallIssue]
        except duckdb.NotImplementedException:
            duckdb.remove_function(udf["name"])  # pyright: ignore[reportUnusedCallResult, reportArgumentType, reportInvalidTypeArguments]
            duckdb.create_function(**udf)  # pyright: ignore[reportCallIssue]
