from collections import OrderedDict, defaultdict
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
from pydantic import BaseModel, ConfigDict, Field, validate_call
from structlog import get_logger

logger = get_logger(__name__)


class InterpolationConfig(BaseModel):
    method: Literal["interp"] = "interp"

    model_config = ConfigDict(extra="forbid")


class AsOfJoinConfig(BaseModel):
    method: Literal["asof"] = "asof"
    strategy: AsofJoinStrategy = "backward"
    tolerance: str | int | float | timedelta | None = None

    model_config = ConfigDict(extra="forbid")


ColumnAlignmentConfig = InterpolationConfig | AsOfJoinConfig


class AlignmentConfig(BaseModel):
    key: str
    columns: OrderedDict[str, ColumnAlignmentConfig] = Field(
        default_factory=OrderedDict
    )

    model_config = ConfigDict(extra="forbid")


type Fields = OrderedDict[str, AlignmentConfig | Fields]


@final
class DataFrameAligner:
    @validate_call
    def __init__(self, *, fields: Fields, separator: str = "/") -> None:
        self._fields = fields
        self._separator = separator

    @cached_property
    def _fully_qualified_fields(self) -> PyTree[AlignmentConfig]:
        def fqn(path: tuple[str, ...], cfg: AlignmentConfig) -> AlignmentConfig:
            key = self._separator.join((*path, cfg.key))
            columns = OrderedDict({
                self._separator.join((*path, k)): v for k, v in cfg.columns.items()
            })

            return AlignmentConfig(key=key, columns=columns)

        return tree_map_with_path(fqn, self._fields)  # ty: ignore[invalid-argument-type]

    def __call__(
        self, input: PyTree[pl.DataFrame] | None = None, **kwargs: PyTree[pl.DataFrame]
    ) -> pl.DataFrame:
        match input, kwargs:
            case [None, _]:
                input = kwargs  # ty: ignore[invalid-assignment]

            case [_, {}]:
                pass

            case _:
                msg = "either `input` or `kwargs` must be specified"
                raise ValueError(msg)

        result = self._build(input)  # ty: ignore[invalid-argument-type]
        logger.debug(
            "aligned dataframes",
            length={"input": tree_map(len, input), "result": len(result)},  # ty: ignore[invalid-argument-type]
        )

        return result

    def _build(self, input: PyTree[pl.DataFrame]) -> pl.DataFrame:
        fields = self._fully_qualified_fields
        accessors = tree_accessors(fields)
        accessor, *accessors_rest = accessors
        left_on = accessor(fields).key

        def get_df(accessor: PyTreeAccessor, cfg: AlignmentConfig) -> pl.DataFrame:
            return (
                accessor(input)
                .rename(lambda col: self._separator.join((*accessor.path, col)))
                .sort(cfg.key)
            )

        dfs = tree_map_with_accessor(get_df, fields)
        df: pl.DataFrame = accessor(dfs)

        for accessor in accessors_rest:
            other: pl.DataFrame = accessor(dfs)
            align_config: AlignmentConfig = accessor(fields)
            key = align_config.key
            df_columns = df.columns
            asof_columns: defaultdict[
                tuple[AsofJoinStrategy, str | int | float | timedelta | None], list[str]
            ] = defaultdict(list)
            interp_columns: list[str] = []

            for column, config in align_config.columns.items():
                match config:
                    case AsOfJoinConfig(strategy=strategy, tolerance=tolerance):
                        asof_columns[strategy, tolerance].append(column)

                    case InterpolationConfig():
                        if key == column:
                            logger.error(msg := "cannot interpolate key")

                            raise ValueError(msg)

                        interp_columns.append(column)

            for (strategy, tolerance), columns in asof_columns.items():
                right_on = uuid4().hex
                df = df.join_asof(
                    other=other.select(pl.col(key).alias(right_on), *columns),
                    left_on=left_on,
                    right_on=right_on,
                    strategy=strategy,
                    tolerance=tolerance,
                ).drop(right_on)

            if interp_columns:
                df = (
                    # take a union of timestamps
                    df
                    .join(
                        other.select(key, *interp_columns),
                        how="full",
                        left_on=left_on,
                        right_on=key,
                        coalesce=True,
                    )
                    # interpolate
                    .with_columns(pl.col(interp_columns).interpolate_by(left_on))
                    # narrow back to original ref col
                    .join(df.select(left_on), on=left_on, how="semi")
                    .sort(left_on)
                )

            df = df.select(*df_columns, *align_config.columns)

        return df
