from collections.abc import Callable
from typing import Any, Literal, TypedDict

import duckdb.typing

DuckDbPyTypeStr = Literal[
    *[
        attr
        for attr in dir(duckdb.typing)
        if isinstance(getattr(duckdb.typing, attr), duckdb.typing.DuckDBPyType)
    ]
]


class DuckDbUdfKwargs(TypedDict):
    name: str
    function: Callable[..., Any]
    parameters: list[DuckDbPyTypeStr] | None  # pyright: ignore[reportInvalidTypeForm]
    return_type: DuckDbPyTypeStr | None  # pyright: ignore[reportInvalidTypeForm]
