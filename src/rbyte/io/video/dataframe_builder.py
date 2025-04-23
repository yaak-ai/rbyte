from os import PathLike
from pathlib import Path
from typing import Literal, final

import polars as pl
from polars.datatypes import IntegerType
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from torchcodec.decoders import VideoDecoder

logger = get_logger(__name__)


type Fields = dict[Literal["frame_idx"], InstanceOf[IntegerType]]


@final
class VideoDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(self, fields: Fields) -> None:
        self._fields = fields

    def __call__(self, path: PathLike[str]) -> pl.DataFrame:
        with bound_contextvars(path=path):
            result = self._build(path)
            logger.debug("built dataframe", length=len(result))

            return result

    def _build(self, path: PathLike[str]) -> pl.DataFrame:
        decoder = VideoDecoder(Path(path).resolve().as_posix())
        match num_frames := decoder.metadata.num_frames:
            case int():
                data = pl.arange(num_frames, eager=True)

            case None:
                raise RuntimeError

        return pl.DataFrame(data=data, schema=self._fields)  # pyright: ignore[reportArgumentType]
