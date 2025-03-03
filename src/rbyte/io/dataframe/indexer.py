from functools import partial
from typing import final

import polars as pl
from optree import PyTree, tree_map
from pydantic import validate_call


@final
class DataFrameIndexer:
    __name__ = __qualname__

    @validate_call
    def __init__(self, name: str) -> None:
        self._fn = partial(pl.DataFrame.with_row_index, name=name)

    def __call__(self, input: PyTree[pl.DataFrame]) -> PyTree[pl.DataFrame]:
        return tree_map(self._fn, input)
