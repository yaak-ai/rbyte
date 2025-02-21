from typing import final

import polars as pl
from optree import PyTree, tree_leaves, tree_map_with_path
from polars._typing import ConcatMethod
from pydantic import validate_call


@final
class DataFrameConcater:
    __name__ = __qualname__

    @validate_call
    def __init__(
        self, method: ConcatMethod = "horizontal", separator: str | None = None
    ) -> None:
        self._method: ConcatMethod = method
        self._separator = separator

    def __call__(self, input: PyTree[pl.DataFrame]) -> pl.DataFrame:
        if (sep := self._separator) is not None:
            input = tree_map_with_path(
                lambda path, df: df.rename(  # pyright: ignore[reportUnknownArgumentType,reportUnknownLambdaType, reportUnknownMemberType]
                    lambda col: f"{sep.join([*path, col])}"  # pyright: ignore[reportUnknownArgumentType,reportUnknownLambdaType]
                ),
                input,
            )

        return pl.concat(tree_leaves(input), how=self._method, rechunk=True)
