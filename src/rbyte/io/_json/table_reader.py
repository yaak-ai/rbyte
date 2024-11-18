import json
from collections.abc import Hashable, Mapping, Sequence
from enum import StrEnum, unique
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import override

import polars as pl
from optree import PyTree, tree_map
from polars._typing import PolarsDataType
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from pydantic import Field
from xxhash import xxh3_64_intdigest as digest

from rbyte.config.base import BaseModel, HydraConfig
from rbyte.io.table.base import TableReader
from rbyte.io.table.transforms.base import TableTransform
from rbyte.utils.dataframe.misc import unnest_all


@unique
class SpecialField(StrEnum):
    idx = "_idx_"


class Config(BaseModel):
    fields: Mapping[str, Mapping[str, HydraConfig[PolarsDataType] | None]]
    transforms: Sequence[HydraConfig[TableTransform]] = Field(default=())


class JsonTableReader(TableReader, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config: Config = Config.model_validate(kwargs)

    @override
    def read(self, path: PathLike[str]) -> PyTree[pl.DataFrame]:
        dfs: Mapping[str, pl.DataFrame] = {}

        for k, series in (
            pl.read_json(Path(path)).select(self._fields).to_dict().items()
        ):
            df_schema = {
                name: dtype
                for name, dtype in self._fields[k].items()
                if dtype is not None
            }
            df = pl.DataFrame(series).lazy().explode(k).unnest(k)
            df = (
                df.select(unnest_all(df.collect_schema()))
                .select(self._config.fields[k].keys() - set(SpecialField))
                .cast(df_schema)  # pyright: ignore[reportArgumentType]
                .collect()
            )

            for transform in self._transforms:
                df = transform(df)

            if (idx_name := SpecialField.idx) in (df_schema := self._fields[k]):
                df = df.with_row_index(idx_name).cast({
                    idx_name: df_schema[idx_name] or pl.UInt32
                })

            dfs[k] = df

        return dfs  # pyright: ignore[reportReturnType]

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)

    @cached_property
    def _fields(self) -> Mapping[str, Mapping[str, PolarsDataType | None]]:
        return tree_map(HydraConfig.instantiate, self._config.fields)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType, reportReturnType]

    @cached_property
    def _transforms(self) -> Sequence[TableTransform]:
        return tuple(transform.instantiate() for transform in self._config.transforms)
