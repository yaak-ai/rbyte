from typing import TYPE_CHECKING

import utm
from duckdb.typing import BLOB
from shapely import Point, wkb

from .base import DuckDbUdfKwargs

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def wgs84_to_utm(point_wkb: bytes) -> bytes:
    point: BaseGeometry = wkb.loads(point_wkb)
    x, y, _, _ = utm.from_latlon(point.y, point.x)  # type: ignore[reportUnknownMemberType]
    return wkb.dumps(Point(x, y))  # type: ignore[reportUnknownMemberType]


Wgs84ToUtm = DuckDbUdfKwargs(
    name="ST_Wgs84ToUtm", function=wgs84_to_utm, parameters=[BLOB], return_type=BLOB
)
