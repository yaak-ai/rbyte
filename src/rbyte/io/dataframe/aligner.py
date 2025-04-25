from collections import OrderedDict
from datetime import timedelta
from functools import cached_property
from typing import Literal, final
from uuid import uuid4

import polars as pl
from optree import (
    PyTree,
    PyTreeAccessor,
    tree_accessors,
    tree_map,
    tree_map_with_accessor,
    tree_map_with_path,
)
from polars._typing import AsofJoinStrategy
from pydantic import Field, validate_call
from structlog import get_logger

from rbyte.config.base import BaseModel

logger = get_logger(__name__)


class InterpColumnAlignConfig(BaseModel):
    method: Literal["interp"] = "interp"


class AsofColumnAlignConfig(BaseModel):
    method: Literal["asof"] = "asof"
    strategy: AsofJoinStrategy = "backward"
    tolerance: str | int | float | timedelta | None = None


type ColumnAlignConfig = InterpColumnAlignConfig | AsofColumnAlignConfig


class AlignConfig(BaseModel):
    key: str
    columns: OrderedDict[str, ColumnAlignConfig] = Field(default_factory=OrderedDict)


type Fields = AlignConfig | OrderedDict[str, Fields]


@final
class DataFrameAligner:
    __name__ = __qualname__

    @validate_call
    def __init__(self, *, fields: Fields, separator: str = "/") -> None:
        self._fields = fields
        self._separator = separator

    @cached_property
    def _fully_qualified_fields(self) -> PyTree[AlignConfig]:
        def fqn(path: tuple[str, ...], cfg: AlignConfig) -> AlignConfig:
            key = self._separator.join((*path, cfg.key))
            columns = OrderedDict({
                self._separator.join((*path, k)): v for k, v in cfg.columns.items()
            })

            return AlignConfig(key=key, columns=columns)

        return tree_map_with_path(fqn, self._fields)  # pyright: ignore[reportArgumentType]

    def __call__(self, input: PyTree[pl.DataFrame]) -> pl.DataFrame:
        result = self._build(input)
        logger.debug(
            "aligned dataframes",
            length={"input": tree_map(len, input), "result": len(result)},
        )

        return result

    def _build(self, input: PyTree[pl.DataFrame]) -> pl.DataFrame:
        fields = self._fully_qualified_fields
        accessors = tree_accessors(fields)
        accessor, *accessors_rest = accessors
        left_on = accessor(fields).key

        def get_df(accessor: PyTreeAccessor, cfg: AlignConfig) -> pl.DataFrame:
            return (
                accessor(input)
                .rename(lambda col: self._separator.join((*accessor.path, col)))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
                .sort(cfg.key)
            )

        dfs = tree_map_with_accessor(get_df, fields)
        df: pl.DataFrame = accessor(dfs)

        for accessor in accessors_rest:
            other: pl.DataFrame = accessor(dfs)
            align_config: AlignConfig = accessor(fields)
            key = align_config.key

            for column, config in align_config.columns.items():
                match config:
                    case AsofColumnAlignConfig(strategy=strategy, tolerance=tolerance):
                        right_on = key if key == column else uuid4().hex

                        df = df.join_asof(
                            other=other.select({key, column}).rename({key: right_on}),
                            left_on=left_on,
                            right_on=right_on,
                            strategy=strategy,
                            tolerance=tolerance,
                        ).drop({right_on} - {key})

                    case InterpColumnAlignConfig():
                        if key == column:
                            logger.error(msg := "cannot interpolate key")

                            raise ValueError(msg)

                        right_on = key

                        df = (
                            # take a union of timestamps
                            df.join(
                                other.select(right_on, column),
                                how="full",
                                left_on=left_on,
                                right_on=right_on,
                                coalesce=True,
                            )
                            # interpolate
                            .with_columns(pl.col(column).interpolate_by(left_on))
                            # narrow back to original ref col
                            .join(df.select(left_on), on=left_on, how="semi")
                            .sort(left_on)
                        )

        return df
