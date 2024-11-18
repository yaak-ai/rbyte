import json
from collections.abc import Hashable
from typing import override

import polars as pl
from optree import PyTree, tree_leaves, tree_map_with_path
from polars._typing import ConcatMethod
from xxhash import xxh3_64_intdigest as digest

from rbyte.config import BaseModel

from .base import TableMerger


class Config(BaseModel):
    separator: str | None = None
    method: ConcatMethod = "horizontal"


class TableConcater(TableMerger, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config: Config = Config.model_validate(kwargs)

    @override
    def merge(self, src: PyTree[pl.DataFrame]) -> pl.DataFrame:
        if (sep := self._config.separator) is not None:
            src = tree_map_with_path(
                lambda path, df: df.rename(  # pyright: ignore[reportUnknownArgumentType,reportUnknownLambdaType, reportUnknownMemberType]
                    lambda col: f"{sep.join([*path, col])}"  # pyright: ignore[reportUnknownArgumentType,reportUnknownLambdaType]
                ),
                src,
            )

        return pl.concat(tree_leaves(src), how=self._config.method, rechunk=True)

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)
