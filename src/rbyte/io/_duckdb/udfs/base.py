from collections.abc import Callable
from typing import Any, TypedDict

from duckdb.typing import DuckDBPyType
from pydantic import InstanceOf


class DuckDbUdfKwargs(TypedDict):
    name: str
    function: Callable[..., Any]
    parameters: list[InstanceOf[DuckDBPyType]] | None
    return_type: InstanceOf[DuckDBPyType] | None
