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
    tree_map_with_accessor,
    tree_map_with_path,
)
from polars._typing import AsofJoinStrategy
from pydantic import Field, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars

from rbyte.config.base import BaseModel

logger = get_logger(__name__)


class InterpColumnMergeConfig(BaseModel):
    method: Literal["interp"] = "interp"


class AsofColumnMergeConfig(BaseModel):
    method: Literal["asof"] = "asof"
    strategy: AsofJoinStrategy = "backward"
    tolerance: str | int | float | timedelta | None = None


type ColumnMergeConfig = InterpColumnMergeConfig | AsofColumnMergeConfig


class MergeConfig(BaseModel):
    key: str
    columns: OrderedDict[str, ColumnMergeConfig] = Field(default_factory=OrderedDict)


type Fields = MergeConfig | OrderedDict[str, "Fields"]


@final
class DataFrameAligner:
    __name__ = __qualname__

    @validate_call
    def __init__(self, *, fields: Fields, separator: str = "/") -> None:
        self._fields = fields
        self._separator = separator

    @cached_property
    def _fully_qualified_fields(self) -> PyTree[MergeConfig]:
        def fqn(path: tuple[str, ...], cfg: MergeConfig) -> MergeConfig:
            key = self._separator.join((*path, cfg.key))
            columns = OrderedDict({
                self._separator.join((*path, k)): v for k, v in cfg.columns.items()
            })

            return MergeConfig(key=key, columns=columns)

        return tree_map_with_path(fqn, self._fields)  # pyright: ignore[reportArgumentType]

    def __call__(self, input: PyTree[pl.DataFrame]) -> pl.DataFrame:
        fields = self._fully_qualified_fields

        def get_df(accessor: PyTreeAccessor, cfg: MergeConfig) -> pl.DataFrame:
            return (
                accessor(input)
                .rename(lambda col: self._separator.join((*accessor.path, col)))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
                .sort(cfg.key)
            )

        dfs = tree_map_with_accessor(get_df, fields)
        accessor, *accessors_rest = tree_accessors(fields)
        df: pl.DataFrame = accessor(dfs)
        left_on = accessor(fields).key

        for accessor in accessors_rest:
            other: pl.DataFrame = accessor(dfs)
            merge_config: MergeConfig = accessor(fields)
            key = merge_config.key

            for column, config in merge_config.columns.items():
                df_height_pre = df.height

                with bound_contextvars(key=key, column=column, config=config):
                    match config:
                        case AsofColumnMergeConfig(
                            strategy=strategy, tolerance=tolerance
                        ):
                            right_on = key if key == column else uuid4().hex

                            df = (
                                df.join_asof(
                                    other=other.select({key, column}).rename({
                                        key: right_on
                                    }),
                                    left_on=left_on,
                                    right_on=right_on,
                                    strategy=strategy,
                                    tolerance=tolerance,
                                )
                                .drop_nulls(column)
                                .drop({right_on} - {key})
                            )

                        case InterpColumnMergeConfig():
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
                            ).drop_nulls(column)

                logger.debug(
                    "merged", column=column, height=f"{df_height_pre}->{df.height}"
                )

        return df
