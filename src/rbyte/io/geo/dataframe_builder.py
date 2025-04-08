from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TypedDict, Unpack, final

import polars as pl
import polars_st as st
from pydantic import validate_call


class _Kwargs(TypedDict, total=False):
    layer: int | str | None
    encoding: str | None
    columns: Sequence[str] | None
    read_geometry: bool
    force_2d: bool
    skip_features: int
    max_features: int | None
    where: str | None
    bbox: tuple[float, float, float, float] | None
    fids: Sequence[int] | None
    sql: str | None
    sql_dialect: str | None
    return_fids: bool


@final
class GeoDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(self, *, srid: int | None = None, **kwargs: Unpack[_Kwargs]) -> None:
        self._srid = srid
        self._kwargs = kwargs

    def __call__(self, path: PathLike[str]) -> pl.DataFrame:
        gdf = st.read_file(Path(path), **self._kwargs)
        if self._srid is not None:
            gdf = gdf.with_columns(st.geom().st.set_srid(self._srid))

        return gdf
