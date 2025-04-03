from functools import cached_property
from typing import final
from uuid import uuid4

import polars as pl
import polars_st as st
from pydantic import BaseModel, validate_call
from structlog import get_logger

logger = get_logger(__name__)


@final
class WaypointBuilder:
    __name__ = __qualname__

    class Columns(BaseModel):
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
        srid = (
            lf.select(st.geom(self._columns.points).st.srid().unique()).collect().item()
        )
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
            .agg(
                st.geom(self._columns.points)
                .st.multipoint()
                .st.set_srid(srid)
                .alias(self._columns.output)
            )
            .join(lf, on=self._index_column, how="left")
            .drop(self._index_column)
            .collect()
            .head(-(self._length - 1))
        )


@final
class WaypointNormalizer:
    __name__ = __qualname__

    class Columns(BaseModel):
        ego: tuple[str, str]
        waypoints: str
        heading: str
        output: str

    @validate_call
    def __init__(self, *, columns: Columns, srid: int | None = None) -> None:
        self._columns = columns
        self._srid = srid

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._normalize(input)

    @cached_property
    def _ego_column(self) -> str:
        return uuid4().hex

    def _normalize(self, input: pl.DataFrame) -> pl.DataFrame:
        lf = input.lazy()
        srid = (
            lf.select(st.geom(self._columns.waypoints).st.srid().unique())
            .collect()
            .item()
        )
        lf = lf.with_columns(
            st.from_xy(*self._columns.ego).st.set_srid(srid).alias(self._ego_column),
            st.geom(self._columns.waypoints).alias(self._columns.output),
        )

        if self._srid is not None:
            lf = lf.with_columns(
                st.geom(self._ego_column, self._columns.output).st.to_srid(self._srid)
            )

        lf = (
            lf.with_columns(
                st.geom(self._columns.output)
                .st.translate(
                    x=-st.geom(self._ego_column).st.x(),
                    y=-st.geom(self._ego_column).st.y(),
                )
                .st.rotate(angle=pl.col(self._columns.heading), origin=(0.0, 0.0))
            )
            .drop(self._ego_column)
            .with_columns(
                st.geom(self._columns.waypoints, self._columns.output)
                .st.parts()
                .list.eval(pl.concat_arr(st.element().st.x(), st.element().st.y()))
            )
        )

        return lf.collect()
