from collections.abc import Iterable
from typing import Any, final

import polars as pl


@final
class SampleAggregator:
    __name__ = __qualname__

    def __init__(self, input_id_column: str) -> None:
        self._input_id_column = input_id_column

    def __call__(
        self, input_ids: Iterable[Any], samples: Iterable[pl.DataFrame]
    ) -> pl.DataFrame:
        input_id_enum = pl.Enum(input_ids)
        return pl.concat([
            df.select(
                pl.lit(input_id).cast(input_id_enum).alias(self._input_id_column),
                pl.col(sorted(df.collect_schema().names())),
            )
            for input_id, df in zip(input_ids, samples, strict=True)
        ])
