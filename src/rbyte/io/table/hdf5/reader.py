import json
from collections.abc import Hashable, Mapping
from enum import StrEnum, unique
from functools import cached_property
from os import PathLike
from typing import Any, cast, override

import numpy.typing as npt
import polars as pl
from h5py import Dataset, File, Group
from polars._typing import PolarsDataType
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from pydantic import ConfigDict, ImportString
from xxhash import xxh3_64_intdigest as digest

from rbyte.config import BaseModel
from rbyte.config.base import HydraConfig
from rbyte.io.table.base import TableReaderBase


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields: Mapping[
        str,
        Mapping[str, HydraConfig[PolarsDataType] | ImportString[PolarsDataType] | None],
    ]


@unique
class SpecialFields(StrEnum):
    idx = "_idx_"


class Hdf5TableReader(TableReaderBase, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config = Config.model_validate(kwargs)

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)

    @override
    def read(self, path: PathLike[str]) -> Mapping[str, pl.DataFrame]:
        dfs: Mapping[str, pl.DataFrame] = {}

        with File(path) as f:
            for group_key, schema in self.schemas.items():
                match group := f[group_key]:
                    case Group():
                        series: list[pl.Series] = []
                        for name, dtype in schema.items():
                            match name:
                                case SpecialFields.idx:
                                    pass

                                case _:
                                    match dataset := group[name]:
                                        case Dataset():
                                            values = cast(npt.NDArray[Any], dataset[:])
                                            series.append(
                                                pl.Series(
                                                    name=name,
                                                    values=values,
                                                    dtype=dtype,
                                                )
                                            )

                                        case _:
                                            raise NotImplementedError

                        df = pl.DataFrame(data=series)  # pyright: ignore[reportGeneralTypeIssues]
                        if (idx_name := SpecialFields.idx) in schema:
                            df = df.with_row_index(idx_name).cast({
                                idx_name: schema[idx_name] or pl.UInt32
                            })

                        dfs[group_key] = df

                    case _:
                        raise NotImplementedError

        return dfs

    @cached_property
    def schemas(self) -> Mapping[str, Mapping[str, PolarsDataType | None]]:
        return {
            group_key: {
                dataset_key: leaf.instantiate()
                if isinstance(leaf, HydraConfig)
                else leaf
                for dataset_key, leaf in fields.items()
            }
            for group_key, fields in self._config.fields.items()
        }