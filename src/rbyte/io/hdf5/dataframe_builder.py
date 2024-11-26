from collections.abc import Mapping, Sequence
from os import PathLike
from typing import cast, final

import numpy.typing as npt
import polars as pl
from h5py import Dataset, File
from optree import PyTree, tree_map, tree_map_with_path
from polars._typing import PolarsDataType  # noqa: PLC2701
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from pydantic import ConfigDict, validate_call

type Fields = Mapping[str, PolarsDataType | None] | Mapping[str, "Fields"]


@final
class Hdf5DataFrameBuilder:
    __name__ = __qualname__

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, fields: Fields) -> None:
        self._fields = fields

    def __call__(self, path: PathLike[str]) -> PyTree[pl.DataFrame]:
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

            series = tree_map_with_path(build_series, self._fields, none_is_leaf=True)  # pyright: ignore[reportArgumentType]

            return tree_map(
                pl.DataFrame,
                series,
                is_leaf=lambda obj: isinstance(obj, dict)
                and all(isinstance(v, pl.Series) or v is None for v in obj.values()),  # pyright: ignore[reportUnknownVariableType]
            )
