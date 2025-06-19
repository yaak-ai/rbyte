from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, final

import duckdb
import polars as pl
from .udf import functions_to_register




@final
class DuckDbDataFrameBuilder:
    __name__ = __qualname__

    def __call__(self, *, query: str, path: PathLike[str]) -> pl.DataFrame:
        _ = duckdb.create_function(
            "ST_TransformUtmFromLatLon", st_transform_utm_from_latlon, [BLOB], BLOB
        )

        return duckdb.sql(query=query.format(path=path)).pl()


    
if __name__ == "__main__":
    query = """
            LOAD spatial;
    SET TimeZone = 'UTC';
    SELECT TO_TIMESTAMP(timestamp)::TIMESTAMP as timestamp,
           heading,
           geom as geom,
           ST_AsWKB(geom) as geom_wkb,
           ST_X(ST_Transform(geom, 'EPSG:4326', 'EPSG:3857', always_xy := True)) AS x,
           ST_Y(ST_Transform(geom, 'EPSG:4326', 'EPSG:3857', always_xy := True)) AS y,
           ST_X(geom) AS lon,
           ST_Y(geom) AS lat,
           ST_TransformUtmFromLatLon(geom) AS geom_identity,
    FROM ST_Read('{path}')
    """
    path = "/nasa/drives/yaak/data/Niro104-HQ/2022-12-23--10-41-24/waypoints.json"

    builder = DuckDbDataFrameBuilder()
    df = builder(query=query, path=Path(path))
    print(df.head())
