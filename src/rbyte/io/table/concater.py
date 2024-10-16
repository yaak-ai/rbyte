import json
from collections.abc import Hashable, Mapping
from typing import override

import polars as pl
from polars._typing import ConcatMethod
from xxhash import xxh3_64_intdigest as digest

from rbyte.config import BaseModel
from rbyte.io.table.base import TableMergerBase


class Config(BaseModel):
    separator: str | None = None
    method: ConcatMethod = "horizontal"


class TableConcater(TableMergerBase, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config = Config.model_validate(kwargs)

    @override
    def merge(self, src: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
        if (separator := self._config.separator) is not None:
            src = {
                k: df.select(pl.all().name.prefix(f"{k}{separator}"))
                for k, df in src.items()
            }

        return pl.concat(src.values(), how=self._config.method, rechunk=True)

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)
