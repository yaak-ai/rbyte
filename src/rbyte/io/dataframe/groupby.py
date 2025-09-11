from collections.abc import Mapping, Sequence
from datetime import timedelta
from functools import cached_property
from typing import final
from uuid import uuid4

import polars as pl
from polars._typing import ClosedInterval, Label, StartBy
from pydantic import PositiveInt, validate_call
from structlog import get_logger

logger = get_logger(__name__)


@final
class DataFrameGroupByDynamic:
    """
    Build samples using `polars.DataFrame.group_by_dynamic`.
    """

    __name__ = __qualname__  # ty: ignore[unresolved-reference]

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        index_column: str,
        every: str | timedelta,
        period: str | timedelta | None = None,
        include_boundaries: bool = False,
        closed: ClosedInterval = "left",
        label: Label = "datapoint",
        start_by: StartBy = "datapoint",
        group_by: str | Sequence[str] | None = None,
        gather_every: PositiveInt | Mapping[str, PositiveInt] | None = None,
    ) -> None:
        self._index_column_name = index_column
        self._index_column = pl.col(index_column)
        self._every = every
        self._period = period
        self._include_boundaries = include_boundaries
        self._closed: ClosedInterval = closed
        self._label: Label = label
        self._group_by = group_by
        self._start_by: StartBy = start_by

        match gather_every:
            case None:
                self._agg = pl.all()

            case int():
                self._agg = pl.all().gather_every(gather_every)

            case Mapping():
                self._agg = [
                    pl.exclude(gather_every.keys()),
                    *(pl.col(k).gather_every(v) for k, v in gather_every.items()),
                ]

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        result = self._build(input)
        logger.debug(
            "grouped", index_column=self._index_column_name, length=len(result)
        )

        return result

    @cached_property
    def _index_column_tmp(self) -> str:
        return uuid4().hex

    def _build(self, input: pl.DataFrame) -> pl.DataFrame:
        return (
            input.lazy()
            .sort(self._index_column)
            .with_columns(self._index_column.alias(self._index_column_tmp))
            .group_by_dynamic(
                index_column=self._index_column_tmp,
                every=self._every,
                period=self._period,
                closed=self._closed,
                label=self._label,
                start_by=self._start_by,
                group_by=self._group_by,
            )
            .agg(self._agg)
            .drop(self._index_column_tmp)
            .collect()
        )
