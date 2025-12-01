import datetime
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, final, override

import duckdb
import more_itertools as mit
import polars as pl
from duckdb import DuckDBPyConnection, Statement, StatementType
from pydantic import AfterValidator, InstanceOf, validate_call
from structlog import get_logger

logger = get_logger(__name__)


def _validate_query(query: str) -> str:
    match duckdb.extract_statements(query):
        case [Statement(type=StatementType.SELECT)]:  # ty: ignore[unresolved-attribute]
            pass

        case _:
            logger.debug(msg := "invalid query", expected="SELECT ...", actual=query)
            raise ValueError(msg)

    return query


# https://duckdb.org/docs/stable/clients/python/conversion#object-conversion-python-object-to-duckdb
DuckDBQueryParameter = (
    str
    | Annotated[Path, AfterValidator(Path.as_posix)]
    | bool
    | datetime.timedelta
    | None
)


@final
class DuckDBDataFrameQuery:
    __name__ = __qualname__  # ty: ignore[unresolved-reference]

    @validate_call
    def __init__(
        self,
        *,
        query: Annotated[str, AfterValidator(_validate_query)],
        config: dict[str, str | bool | int | float | list[str]] | None = None,
        extensions: Sequence[str] | None = None,
    ) -> None:
        self._query = query
        self._config = config or {}
        self._extensions = extensions or ()
        with duckdb.connect(config=self._config) as con:
            for extension in self._extensions:
                con.install_extension(extension)

        self._thread_local = threading.local()

    @property
    def con(self) -> DuckDBPyConnection:
        if not hasattr(self._thread_local, "con"):
            self._thread_local.con = duckdb.connect(config=self._config)
            for extension in self._extensions:
                self._thread_local.con.load_extension(extension)

        return self._thread_local.con

    @validate_call
    def __call__(
        self, **kwargs: InstanceOf[pl.DataFrame] | DuckDBQueryParameter
    ) -> pl.DataFrame:
        parameters, views = mit.partition(
            lambda kv: isinstance(kv[1], pl.DataFrame), kwargs.items()
        )

        with register_views(self.con, views) as con:
            return con.execute(self._query, dict(parameters)).pl()

    @override
    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.pop("_thread_local", None)
        return state

    @override
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._thread_local = threading.local()


@contextmanager
def register_views(
    con: DuckDBPyConnection, views: Iterable[tuple[str, object]]
) -> Iterator[DuckDBPyConnection]:
    registered = []
    try:
        for name, obj in views:
            con.register(name, obj)
            registered.append(name)

        yield con

    finally:
        for name in reversed(registered):
            con.unregister(name)
