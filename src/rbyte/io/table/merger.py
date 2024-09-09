import json
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from functools import cached_property
from operator import itemgetter
from typing import Annotated, Literal, Self, override

import more_itertools as mit
import polars as pl
from polars.type_aliases import AsofJoinStrategy
from pydantic import StringConstraints, model_validator
from structlog import get_logger
from xxhash import xxh3_64_intdigest as digest

from rbyte.config.base import BaseModel
from rbyte.io.table.base import TableMergerBase

logger = get_logger(__name__)


class RefColumnMergeConfig(BaseModel):
    method: Literal["ref"] = "ref"


class AsofColumnMergeConfig(BaseModel):
    method: Literal["asof"] = "asof"
    tolerance: Annotated[
        str, StringConstraints(strip_whitespace=True, to_lower=True, pattern=r"\d+ms$")
    ] = "100ms"
    strategy: AsofJoinStrategy = "nearest"


class InterpColumnMergeConfig(BaseModel):
    method: Literal["interp"] = "interp"


MergeConfig = RefColumnMergeConfig | AsofColumnMergeConfig | InterpColumnMergeConfig


class Config(BaseModel):
    merge: OrderedDict[str, Mapping[str, MergeConfig]]
    separator: Annotated[str, StringConstraints(strip_whitespace=True)] = "."

    @model_validator(mode="after")
    def validate_refs(self) -> Self:
        ref_config = RefColumnMergeConfig()
        for k, v in self.columns_by_merge_config.items():
            match v.get(ref_config, None):
                case [_column]:
                    pass

                case _:
                    msg = f"merge `{k}` must have exactly one column with {ref_config}"
                    raise ValueError(msg)

        return self

    @cached_property
    def columns_by_merge_config(
        self,
    ) -> Mapping[str, Mapping[MergeConfig, Sequence[str]]]:
        return {
            k: mit.map_reduce(v.items(), keyfunc=itemgetter(1), valuefunc=itemgetter(0))
            for k, v in self.merge.items()
        }

    @cached_property
    def ref_columns(self) -> Mapping[str, str]:
        return {
            k: mit.one(v[RefColumnMergeConfig()])
            for k, v in self.columns_by_merge_config.items()
        }


class TableMerger(TableMergerBase, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config = Config.model_validate(kwargs)

    def _col_name(self, *args: str) -> str:
        return self._config.separator.join(args)

    @override
    def merge(self, src: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
        dfs = {
            k: src[k]
            .sort(self._config.ref_columns[k])
            .rename(lambda col, k=k: self._col_name(k, col))
            for k in self._config.merge
        }
        k_df_ref = mit.first(self._config.merge.keys())
        df_ref = dfs.pop(k_df_ref)
        df_ref_col_ref = self._col_name(k_df_ref, self._config.ref_columns[k_df_ref])

        logger.debug(
            "merging", merge_ref=f"{k_df_ref}[{self._config.ref_columns[k_df_ref]}]"
        )

        for k_merge, df_merge in dfs.items():
            cols_by_merge_config = self._config.columns_by_merge_config[k_merge]
            df_merge_col_ref = self._col_name(
                k_merge, self._config.ref_columns[k_merge]
            )

            for merge_cfg, _df_merge_cols in cols_by_merge_config.items():
                if isinstance(merge_cfg, RefColumnMergeConfig):
                    continue

                df_merge_cols = tuple(
                    self._col_name(k_merge, col) for col in _df_merge_cols
                )

                df_ref_height_pre = df_ref.height
                match merge_cfg:
                    case AsofColumnMergeConfig(strategy=strategy, tolerance=tolerance):
                        df_ref = df_ref.join_asof(
                            other=df_merge.select(df_merge_col_ref, *df_merge_cols),
                            left_on=df_ref_col_ref,
                            right_on=df_merge_col_ref,
                            strategy=strategy,
                            tolerance=tolerance,
                        ).drop_nulls(df_merge_cols)

                    case InterpColumnMergeConfig():
                        df_ref = (
                            # take a union of timestamps
                            df_ref.join(
                                df_merge.select(df_merge_col_ref, *df_merge_cols),
                                how="full",
                                left_on=df_ref_col_ref,
                                right_on=df_merge_col_ref,
                                coalesce=True,
                            )
                            # interpolate
                            .with_columns(
                                pl.col(df_merge_cols).interpolate_by(df_ref_col_ref)
                            )
                            # narrow back to original ref col
                            .join(
                                df_ref.select(df_ref_col_ref),
                                on=df_ref_col_ref,
                                how="semi",
                            )
                            .sort(df_ref_col_ref)
                        ).drop_nulls(df_merge_cols)

                logger.debug(
                    "merged",
                    merge_rows=f"{df_ref_height_pre}->{df_ref.height}",
                    merge_other=f"{k_merge}[{", ".join(_df_merge_cols)}]",
                )

        return df_ref

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)
