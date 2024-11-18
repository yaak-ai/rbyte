from collections.abc import Callable, Iterable
from typing import cast, override

import numpy.typing as npt
import polars as pl
import rerun as rr
import torch
from jaxtyping import UInt8
from pydantic import FilePath, validate_call
from rerun.components import Blob
from torch import Tensor

from rbyte.config.base import BaseModel
from rbyte.io.base import TensorSource


class RrdFrameSource(TensorSource):
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self,
        path: FilePath,
        *,
        index: str,
        entity_path: str,
        decoder: Callable[[bytes], npt.ArrayLike],
    ) -> None:
        recording = rr.dataframe.load_recording(path)  # pyright: ignore[reportUnknownMemberType]
        view = recording.view(index=index, contents={entity_path: [Blob]})
        reader = view.select(columns=[index, f"{entity_path}:{Blob.__name__}"])

        # WARN: RecordBatchReader does not support random seeking => storing in memory
        self._series: pl.Series = (
            cast(
                pl.DataFrame,
                pl.from_arrow(reader.read_all(), rechunk=True),  # pyright: ignore[reportUnknownMemberType]
            )
            .sort(index)
            .drop(index)
            .select(pl.all().explode())
            .to_series(0)
        )

        self._decoder: Callable[[bytes], npt.ArrayLike] = decoder

    @override
    def __getitem__(self, indexes: Iterable[int]) -> UInt8[Tensor, "b h w c"]:
        arrays = (self._series[i].to_numpy(allow_copy=False) for i in indexes)
        frames_np = map(self._decoder, arrays)
        frames_tch = map(torch.from_numpy, frames_np)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        return torch.stack(list(frames_tch))

    @override
    def __len__(self) -> int:
        return len(self._series)
