from typing import TYPE_CHECKING

import utm
from duckdb.typing import BLOB
from shapely import wkb
from shapely.geometry import Point

from .base import DuckDbUdfKwargs

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def wgs84_to_utm(point_wkb: bytes) -> bytes:
    point: BaseGeometry = wkb.loads(point_wkb)
    lat, lon = point.representative_point().y, point.representative_point().x
    x, y, _, _ = utm.from_latlon(lat, lon)  # type: ignore[reportUnknownMemberType]
    return wkb.dumps(Point(x, y))  # type: ignore[reportUnknownMemberType]


Wgs84ToUtm = DuckDbUdfKwargs(
    name="ST_Wgs84ToUtm", function=wgs84_to_utm, parameters=[BLOB], return_type=BLOB
)
