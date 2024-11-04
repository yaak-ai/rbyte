import json
from collections.abc import Hashable, Mapping, Sequence
from enum import StrEnum, unique
from functools import cached_property
from os import PathLike
from typing import cast, override

import numpy.typing as npt
import polars as pl
from h5py import Dataset, File
from optree import PyTree, tree_map, tree_map_with_path
from polars._typing import PolarsDataType  # noqa: PLC2701
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from xxhash import xxh3_64_intdigest as digest

from rbyte.config import BaseModel
from rbyte.config.base import HydraConfig
from rbyte.io.table.base import TableReader

type Fields = Mapping[str, HydraConfig[PolarsDataType] | None] | Mapping[str, "Fields"]


class Config(BaseModel):
    fields: Fields


Config.model_rebuild()  # pyright: ignore[reportUnusedCallResult]


@unique
class SpecialField(StrEnum):
    idx = "_idx_"


class Hdf5TableReader(TableReader, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config: Config = Config.model_validate(kwargs)

    @override
    def read(self, path: PathLike[str]) -> PyTree[pl.DataFrame]:
        with File(path) as f:

            def build_series(
                path: Sequence[str], dtype: PolarsDataType | None
            ) -> pl.Series | None:
                key = "/".join(path)
                match obj := f.get(key):  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                    case Dataset():
                        values = cast(npt.ArrayLike, obj[:])
                        return pl.Series(values=values, dtype=dtype)

                    case None:
                        return None

                    case _:  # pyright: ignore[reportUnknownVariableType]
                        raise NotImplementedError

            series = tree_map_with_path(build_series, self._fields, none_is_leaf=True)

            dfs = tree_map(
                pl.DataFrame,
                series,
                is_leaf=lambda obj: isinstance(obj, dict)
                and all(isinstance(v, pl.Series) or v is None for v in obj.values()),  # pyright: ignore[reportUnknownVariableType]
            )

            def maybe_add_index(
                df: pl.DataFrame, schema: Mapping[str, PolarsDataType | None]
            ) -> pl.DataFrame:
                match schema:
                    case {SpecialField.idx: dtype}:
                        return df.select(
                            pl.int_range(pl.len(), dtype=dtype or pl.UInt32).alias(  # pyright: ignore[reportArgumentType]
                                SpecialField.idx
                            ),
                            pl.exclude(SpecialField.idx),
                        )

                    case _:
                        return df

            return tree_map(maybe_add_index, dfs, self._fields)

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)

    @cached_property
    def _fields(self) -> PyTree[PolarsDataType | None]:
        return tree_map(HydraConfig.instantiate, self._config.fields)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]
