import json
from collections import OrderedDict
from collections.abc import Hashable
from datetime import timedelta
from functools import cached_property
from typing import Annotated, Literal, override
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
from pydantic import Field, StringConstraints
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from xxhash import xxh3_64_intdigest as digest

from rbyte.config.base import BaseModel

from .base import TableMerger

logger = get_logger(__name__)


class InterpColumnMergeConfig(BaseModel):
    method: Literal["interp"] = "interp"


class AsofColumnMergeConfig(BaseModel):
    method: Literal["asof"] = "asof"
    strategy: AsofJoinStrategy = "backward"
    tolerance: str | int | float | timedelta | None = None


ColumnMergeConfig = InterpColumnMergeConfig | AsofColumnMergeConfig


class TableMergeConfig(BaseModel):
    key: str
    columns: OrderedDict[str, ColumnMergeConfig] = Field(default_factory=OrderedDict)


type MergeConfig = TableMergeConfig | OrderedDict[str, "MergeConfig"]


class Config(BaseModel):
    merge: MergeConfig
    separator: Annotated[str, StringConstraints(strip_whitespace=True)] = "/"

    @cached_property
    def merge_fqn(self) -> PyTree[TableMergeConfig]:
        # fully qualified key/column names
        def fqn(path: tuple[str, ...], cfg: TableMergeConfig) -> TableMergeConfig:
            key = self.separator.join((*path, cfg.key))
            columns = OrderedDict({
                self.separator.join((*path, k)): v for k, v in cfg.columns.items()
            })

            return TableMergeConfig(key=key, columns=columns)

        return tree_map_with_path(fqn, self.merge)  # pyright: ignore[reportArgumentType]


class TableAligner(TableMerger, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config: Config = Config.model_validate(kwargs)

    @override
    def merge(self, src: PyTree[pl.DataFrame]) -> pl.DataFrame:
        merge_configs = self._config.merge_fqn

        def get_df(accessor: PyTreeAccessor, cfg: TableMergeConfig) -> pl.DataFrame:
            return (
                accessor(src)
                .rename(lambda col: self._config.separator.join((*accessor.path, col)))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
                .sort(cfg.key)
            )

        dfs = tree_map_with_accessor(get_df, merge_configs)
        accessor, *accessors_rest = tree_accessors(merge_configs)
        df: pl.DataFrame = accessor(dfs)
        left_on = accessor(merge_configs).key

        for accessor in accessors_rest:
            other: pl.DataFrame = accessor(dfs)
            merge_config: TableMergeConfig = accessor(merge_configs)
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

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)
