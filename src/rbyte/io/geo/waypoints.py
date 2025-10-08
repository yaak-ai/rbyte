from dataclasses import dataclass
from functools import cached_property
from typing import final
from uuid import uuid4

import polars as pl
import polars_st as st
from pydantic import validate_call
from structlog import get_logger

logger = get_logger(__name__)


@final
class WaypointBuilder:
    __name__ = __qualname__  # ty: ignore[unresolved-reference]

    @dataclass
    class Columns:
        points: str
        output: str

    @validate_call
    def __init__(self, *, length: int, columns: Columns) -> None:
        self._length = length
        self._columns = columns

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._build(input)

    @cached_property
    def _index_column(self) -> str:
        return uuid4().hex

    def _build(self, input: pl.DataFrame) -> pl.DataFrame:
        lf = input.lazy()
        lf = pl.concat([lf] + [lf.tail(1)] * (self._length - 1)).with_row_index(
            self._index_column
        )

        return (
            lf.rolling(
                self._index_column,
                period=f"{self._length}i",
                offset="0i",
                closed="left",
            )
            .agg(st.geom(self._columns.points).st.collect().alias(self._columns.output))
            .join(lf, on=self._index_column, how="left")
            .drop(self._index_column)
            .collect()
            .head(-(self._length - 1))
        )
