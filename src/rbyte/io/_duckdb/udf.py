
from typing import TYPE_CHECKING

import utm
from duckdb.typing import BLOB
from shapely import wkb
from shapely.geometry import Point

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


__all__ = ["functions_to_register"]

def st_transform_utm_from_latlon(point_wkb: bytes) -> bytes:
    point: BaseGeometry = wkb.loads(point_wkb)
    lat, lon = point.representative_point().y, point.representative_point().x
    x, y, _, _ = utm.from_latlon(lat, lon)  # pyright: ignore[reportUnknownMemberType]
    return wkb.dumps(Point(x, y))  # pyright: ignore[reportUnknownMemberType]


functions_to_register = [
    {
        "name": "ST_TransformUtmFromLatLon",
        "function": st_transform_utm_from_latlon,
        "parameters": [BLOB],
        "return_type": BLOB,
    }
]
