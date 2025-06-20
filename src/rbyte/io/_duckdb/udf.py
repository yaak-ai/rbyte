from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import utm
from shapely import Point, wkb

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


__all__ = ["DuckDbUdf", "udf_list"]


@dataclass
class DuckDbUdf:
    name: str
    function: Callable[..., Any]
    parameters: list[str] | None = None
    return_type: str | None = None


def st_transform_utm_from_wgs84(point_wkb: bytes) -> bytes:
    point: BaseGeometry = wkb.loads(point_wkb)
    lat, lon = point.representative_point().y, point.representative_point().x
    x, y, _, _ = utm.from_latlon(lat, lon)  # type: ignore[reportUnknownMemberType]
    return wkb.dumps(Point(x, y))  # type: ignore[reportUnknownMemberType]


udf_list: list[DuckDbUdf] = [
    DuckDbUdf(
        name="ST_UtmFromWgs84",
        function=st_transform_utm_from_wgs84,
        parameters=["BLOB"],
        return_type="BLOB",
    )
]
