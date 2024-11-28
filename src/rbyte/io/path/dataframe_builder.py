import os
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import final

import parse
import polars as pl
from polars._typing import PolarsDataType  # noqa: PLC2701
from polars.datatypes import (
    DataType,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    DataTypeClass,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)
from pydantic import ConfigDict, validate_call
from structlog import get_logger

logger = get_logger(__name__)


type Fields = Mapping[str, PolarsDataType | None]


@final
class PathDataFrameBuilder:
    __name__ = __qualname__

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, fields: Fields) -> None:
        self._fields = fields

    def __call__(self, path: PathLike[str]) -> pl.DataFrame:
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

        return pl.DataFrame(
            data=(r.named for r in results if isinstance(r, parse.Result)),  # pyright: ignore[reportUnknownMemberType]
            schema=self._fields,
        )
