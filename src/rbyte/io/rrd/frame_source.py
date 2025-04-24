from collections.abc import Callable, Sequence
from typing import cast, final, override

import numpy.typing as npt
import polars as pl
import rerun as rr
import torch
from pydantic import FilePath, validate_call
from rerun.components import Blob
from torch import Tensor

from rbyte.config.base import BaseModel
from rbyte.io.base import TensorSource


@final
class RrdFrameSource(TensorSource[int]):
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self,
        path: FilePath,
        *,
        index: str,
        entity_path: str,
        decoder: Callable[[bytes], npt.ArrayLike],
    ) -> None:
        recording = rr.dataframe.load_recording(path)
        view = recording.view(index=index, contents={entity_path: [Blob]})
        reader = view.select(columns=[index, f"{entity_path}:{Blob.__name__}"])

        # WARN: RecordBatchReader does not support random seeking => storing in memory
        self._series = (
            cast(
                pl.DataFrame,
                pl.from_arrow(reader.read_all(), rechunk=True),  # pyright: ignore[reportUnknownMemberType]
            )
            .sort(index)
            .drop(index)
            .select(pl.all().explode())
            .to_series(0)
        )

        self._decoder = decoder

    def _getitem(self, index: int) -> Tensor:
        array = self._series[index].to_numpy(allow_copy=False)
        array = self._decoder(array)
        return torch.from_numpy(array)  # pyright: ignore[reportUnknownMemberType]

    @override
    def __getitem__(self, indexes: int | Sequence[int]) -> Tensor:
        match indexes:
            case Sequence():
                tensors = map(self._getitem, indexes)

                return torch.stack(list(tensors))

            case int():
                return self._getitem(indexes)

    @override
    def __len__(self) -> int:
        return len(self._series)
