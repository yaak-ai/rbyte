from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal, final

import polars as pl
from polars._typing import PolarsIntegerType  # noqa: PLC2701
from polars.datatypes import (
    IntegerType,  # pyright: ignore[reportUnusedImport] # noqa: F401
)
from pydantic import ConfigDict, validate_call
from video_reader import (
    PyVideoReader,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)

type Fields = Mapping[Literal["frame_idx"], PolarsIntegerType]


@final
class VideoDataFrameBuilder:
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, fields: Fields) -> None:
        self._fields = fields

    def __call__(self, path: PathLike[str]) -> pl.DataFrame:
        vr = PyVideoReader(Path(path).resolve().as_posix())  # pyright: ignore[reportUnknownVariableType]
        info = vr.get_info()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        frame_count = int(info["frame_count"])  # pyright: ignore[reportUnknownArgumentType]
        data = pl.arange(frame_count, eager=True)

        return pl.DataFrame(data=data, schema=self._fields)  # pyright: ignore[reportArgumentType]
