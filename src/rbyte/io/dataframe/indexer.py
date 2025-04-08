from typing import final

import polars as pl
from optree import PyTree, tree_map
from polars.datatypes import DataType, DataTypeClass
from pydantic import InstanceOf, validate_call


@final
class DataFrameIndexer:
    __name__ = __qualname__

    @validate_call
    def __init__(
        self,
        name: str,
        offset: int = 0,
        dtype: InstanceOf[DataType] | InstanceOf[DataTypeClass] | None = None,
    ) -> None:
        self._name = name
        self._offset = offset
        self._dtype = dtype

    def __call__(self, input: PyTree[pl.DataFrame]) -> PyTree[pl.DataFrame]:
        return tree_map(self._index, input)

    def _index(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_row_index(name=self._name, offset=self._offset)
        if self._dtype is not None:
            df = df.cast(dtypes={self._name: self._dtype})

        return df
