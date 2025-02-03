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

        parent = Path(
            os.path.commonpath([path, parser._expression.replace("\\.", ".")])  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        )
        paths = map(Path.as_posix, parent.rglob("*"))
        parsed = map(parser.parse, paths)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        data = (x.named for x in parsed if isinstance(x, parse.Result))  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        return pl.DataFrame(data=data, schema=self._fields)
