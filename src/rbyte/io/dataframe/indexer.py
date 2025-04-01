from functools import partial
from typing import TYPE_CHECKING, final

import polars as pl
from optree import PyTree, tree_map
from pydantic import validate_call

if TYPE_CHECKING:
    from collections.abc import Callable


@final
class DataFrameIndexer:
    __name__ = __qualname__

    @validate_call
    def __init__(self, name: str) -> None:
        indexer_fn = partial(pl.DataFrame.with_row_index, name=name)
        cast_fn = partial(pl.DataFrame.cast, dtypes={name: pl.Int32})

        self._fn: Callable[[pl.DataFrame], pl.DataFrame] = lambda df: cast_fn(
            indexer_fn(df)
        )

    def __call__(self, input: PyTree[pl.DataFrame]) -> PyTree[pl.DataFrame]:
        return tree_map(self._fn, input)
