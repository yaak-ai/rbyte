from collections.abc import Iterable
from functools import partial
from typing import final

import polars as pl
from polars._typing import ConcatMethod
from pydantic import validate_call


@final
class DataFrameConcatenator:
    @validate_call
    def __init__(
        self,
        *,
        key_column: str | None = None,
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
    ) -> None:
        self._key_column = key_column
        self._fn = partial(pl.concat, how=how, rechunk=rechunk, parallel=parallel)

    def __call__(
        self, *, keys: Iterable[str] | None = None, values: Iterable[pl.DataFrame]
    ) -> pl.DataFrame:
        match self._key_column, keys:
            case None, None:
                return self._fn(values)

            case (_, None) | (None, _):
                msg = "`keys` must be provided when `key_column` is specified"
                raise ValueError(msg)

            case _:
                keys = tuple(keys)  # ty: ignore[invalid-argument-type]
                key_enum = pl.Enum(categories=keys)

                return self._fn([
                    v.lazy().with_columns(
                        pl.lit(k).cast(key_enum).alias(self._key_column)  # ty:ignore[invalid-argument-type]
                    )
                    for k, v in zip(keys, values, strict=True)
                ]).collect()
