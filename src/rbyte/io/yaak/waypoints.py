from typing import Annotated, final

import polars as pl
import ymmv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    InstanceOf,
    PositiveInt,
    validate_call,
)
from structlog import get_logger

logger = get_logger(__name__)


type NonNegativeFiniteFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class Search(BaseModel):
    preferred_match_radius_m: NonNegativeFiniteFloat = 100.0
    beam_width: PositiveInt = 64
    max_backtrack_m: NonNegativeFiniteFloat = 5.0
    max_advance_m: NonNegativeFiniteFloat = 200.0

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)


class Weights(BaseModel):
    distance: NonNegativeFiniteFloat = 1.0
    route_heading: NonNegativeFiniteFloat = 5.0
    advance: NonNegativeFiniteFloat = 0.1
    backtrack: NonNegativeFiniteFloat = 25.0

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)


class Config(BaseModel):
    waypoint_offsets_m: tuple[FiniteFloat, ...] = Field(min_length=1)
    search: Search = Field(default_factory=Search)
    weights: Weights = Field(default_factory=Weights)

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    def to_native(self) -> ymmv.RouteMatcherConfig:
        return ymmv.RouteMatcherConfig(
            waypoint_offsets_m=list(self.waypoint_offsets_m),
            search=ymmv.SearchConfig(**self.search.model_dump()),
            weights=ymmv.CostWeights(**self.weights.model_dump()),
        )


@final
class RouteMatchedWaypointGenerator:
    __name__ = __qualname__

    Search = Search
    Weights = Weights
    Config = Config

    @validate_call
    def __init__(self, *, config: Config) -> None:
        self._config = config

    @validate_call
    def __call__(
        self, *, route: InstanceOf[pl.DataFrame], gnss: InstanceOf[pl.DataFrame]
    ) -> pl.DataFrame:
        waypoints = ymmv.generate_route_waypoints(
            config=self._config.to_native(),
            route=route,
            gnss=gnss.select("easting", "northing", "heading"),
        )
        result = gnss.with_columns(waypoints.rename("waypoints/position"))
        logger.debug(
            "generated route waypoints",
            length={
                "gnss": len(gnss),
                "route_features": len(route),
                "waypoints": len(result),
            },
        )

        return result
