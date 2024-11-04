import os
from collections.abc import Mapping
from enum import StrEnum, unique
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import override

import parse
import polars as pl
from optree import PyTree, tree_map
from polars._typing import PolarsDataType
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from structlog import get_logger

from rbyte.config.base import BaseModel, HydraConfig
from rbyte.io.table.base import TableReader

logger = get_logger(__name__)


class Config(BaseModel):
    fields: Mapping[str, HydraConfig[PolarsDataType] | None] = {}


@unique
class SpecialField(StrEnum):
    idx = "_idx_"


class PathTableReader(TableReader):
    def __init__(self, **kwargs: object) -> None:
        self._config: Config = Config.model_validate(kwargs)

    @override
    def read(self, path: PathLike[str]) -> PyTree[pl.DataFrame]:
        parser = parse.compile(Path(path).resolve().as_posix())  # pyright: ignore[reportUnknownMemberType]
        match parser.named_fields, parser.fixed_fields:  # pyright: ignore[reportUnknownMemberType]
            case ([_, *_], []):  # pyright: ignore[reportUnknownVariableType]
                pass

            case (named_fields, fixed_fields):  # pyright: ignore[reportUnknownVariableType]
                logger.error(
                    msg := "parser not supported",
                    named_fields=named_fields,
                    fixed_fields=fixed_fields,
                )
                raise RuntimeError(msg)

        parent = Path(os.path.commonpath([path, parser._expression]))  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        results = (parser.parse(p.as_posix()) for p in parent.rglob("*") if p.is_file())  # pyright: ignore[reportUnknownMemberType]

        df = pl.DataFrame(
            result.named  # pyright: ignore[reportUnknownMemberType]
            for result in results
            if isinstance(result, parse.Result)
        )

        if (idx_name := SpecialField.idx) in self._fields:
            df = df.with_row_index(idx_name).cast({
                idx_name: self._fields[idx_name] or pl.UInt32
            })

        df_schema = {
            name: dtype for name, dtype in self._fields.items() if dtype is not None
        }

        return df.cast(df_schema)  # pyright: ignore[reportArgumentType, reportReturnType]

    @cached_property
    def _fields(self) -> Mapping[str, PolarsDataType | None]:
        return tree_map(HydraConfig.instantiate, self._config.fields)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType, reportReturnType]
