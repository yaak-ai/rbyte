from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum, unique
from typing import cast, override

import numpy.typing as npt
import polars as pl
import rerun as rr
import torch
from jaxtyping import UInt8
from pydantic import ConfigDict, FilePath, validate_call
from rerun.components import Blob, MediaType
from torch import Tensor

from rbyte.io.frame.base import FrameReader


@unique
class Column(StrEnum):
    blob = Blob.__name__
    media_type = MediaType.__name__


class RrdFrameReader(FrameReader):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        path: FilePath,
        *,
        index: str,
        entity_path: str,
        frame_decoders: Mapping[str, Callable[[bytes], npt.ArrayLike]],
    ) -> None:
        recording = rr.dataframe.load_recording(path)  # pyright: ignore[reportUnknownMemberType]
        view = recording.view(index=index, contents={entity_path: (Blob, MediaType)})
        reader = view.select(
            columns=[
                index,
                *(f"{entity_path}:{ct.__name__}" for ct in (Blob, MediaType)),
            ]
        )

        # WARN: RecordBatchReader does not support random seeking => storing in memory
        df = (
            cast(
                pl.DataFrame,
                pl.from_arrow(reader.read_all(), rechunk=True),  # pyright: ignore[reportUnknownMemberType]
            )
            .sort(index)
            .drop(index)
            .select(
                pl.all().explode().name.map(lambda x: x.removeprefix(f"{entity_path}:"))
            )
        )

        self._df = df.cast({
            (col := Column.media_type): pl.Enum(df.select(col).unique().to_series())
        })

        self._frame_decoders = frame_decoders

    @override
    def read(self, indexes: Iterable[int]) -> UInt8[Tensor, "b h w c"]:
        df = self._df[list(indexes)]

        frames_np = (
            self._frame_decoders[media_type](blob.to_numpy(allow_copy=False))
            for media_type, blob in zip(
                df[Column.media_type], df[Column.blob], strict=True
            )
        )

        frames_tch = map(torch.from_numpy, frames_np)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        return torch.stack(list(frames_tch))

    @override
    def get_available_indexes(self) -> Sequence[int]:
        return range(len(self._df))
