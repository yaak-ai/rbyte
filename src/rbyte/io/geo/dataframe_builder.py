from os import PathLike
from typing import final

import polars as pl
import polars_st as st
from pydantic import validate_call


@final
class GeoDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(self, srid: int | None = None) -> None:
        self._srid = srid

    def __call__(self, path: PathLike[str]) -> pl.DataFrame:
        gdf = st.read_file(path)
        if self._srid is not None:
            gdf = gdf.with_columns(st.geom().st.set_srid(self._srid))

        return gdf
