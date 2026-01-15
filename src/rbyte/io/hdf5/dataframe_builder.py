from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Any, final

import polars as pl
from h5py import Dataset, File
from optree import PyTree, tree_map, tree_map_with_path
from polars.datatypes import DataType
from structlog import get_logger
from structlog.contextvars import bound_contextvars

logger = get_logger(__name__)


@final
class Hdf5DataFrameBuilder:
    __name__ = __qualname__

    def __init__(self, fields: Any) -> None:
        # fields is PyTree[Union[InstanceOf[DataType], None]] - recursive type not supported by @validate_call in Python 3.10
        self._fields = fields

    def __call__(self, path: PathLike[str], prefix: str = "/") -> PyTree[pl.DataFrame]:
        with bound_contextvars(path=path, prefix=prefix):
            result = self._build(path, prefix)
            logger.debug("built dataframes", length=tree_map(len, result))

            return result

    def _build(self, path: PathLike[str], prefix: str) -> PyTree[pl.DataFrame]:
        with File(path) as f:

            def build_series(
                path: Sequence[str], dtype: DataType | None
            ) -> pl.Series | None:
                name = "/".join((prefix, *path))
                match obj := f.get(name):
                    case Dataset():
                        return pl.Series(values=obj[:], dtype=dtype)

                    case None:
                        return None

                    case _:
                        raise NotImplementedError

            series = tree_map_with_path(build_series, self._fields, none_is_leaf=True)  # ty: ignore[invalid-argument-type]

            return tree_map(
                pl.DataFrame,
                series,
                is_leaf=lambda obj: (
                    isinstance(obj, dict)
                    and all(isinstance(v, pl.Series) or v is None for v in obj.values())
                ),
            )
