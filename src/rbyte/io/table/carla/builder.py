from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Annotated, override

import polars as pl
from polars import selectors as cs
from pydantic import ConfigDict, StringConstraints, validate_call

from rbyte.io.table.base import TableBuilderBase
from rbyte.io.table.transforms.base import TableTransform
from rbyte.utils.dataframe.misc import unnest_all


class CarlaRecordsTableBuilder(TableBuilderBase):
    RECORD_KEY = "records"

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        index_column: Annotated[
            str, StringConstraints(strip_whitespace=True)
        ] = "frame_idx",
        select: str | frozenset[str] = "*",
        filter: Annotated[str, StringConstraints(strip_whitespace=True)] | None = None,  # noqa: A002
        transforms: Sequence[TableTransform] = (),
    ) -> None:
        super().__init__()

        self._index_column = index_column
        self._select = select
        self._filter = filter
        self._transforms = transforms

    @override
    def build(self, path: PathLike[str]) -> pl.DataFrame:
        df = pl.read_json(Path(path)).explode(self.RECORD_KEY).unnest(self.RECORD_KEY)
        df = (
            df.select(unnest_all(df.collect_schema()))
            .select(self._select)
            # 32 bits ought to be enough for anybody
            .cast({
                cs.by_dtype(pl.Int64): pl.Int32,
                cs.by_dtype(pl.UInt64): pl.UInt32,
                cs.by_dtype(pl.Float64): pl.Float32,
            })
            .sql(f"select * from self where ({self._filter or True})")  # noqa: S608
        )

        for transform in self._transforms:
            df = transform(df)

        return df.with_row_index(self._index_column)
