from collections.abc import Iterable, Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import final, override

import numpy as np
import torch
from numpy.lib.recfunctions import (
    structured_to_unstructured,  # pyright: ignore[reportUnknownVariableType]
)
from pydantic import validate_call
from torch import Tensor

from rbyte.config.base import BaseModel
from rbyte.io.base import TensorSource
from rbyte.utils.tensor import pad_sequence


@final
class NumpyTensorSource(TensorSource):
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self, path: PathLike[str], select: Sequence[str] | None = None
    ) -> None:
        super().__init__()

        self._path = Path(path)
        self._select = select or ...

    @cached_property
    def _path_posix(self) -> str:
        return self._path.resolve().as_posix()

    def _getitem(self, index: object) -> Tensor:
        path = self._path_posix.format(index)
        array = structured_to_unstructured(np.load(path)[self._select])  # pyright: ignore[reportUnknownVariableType]
        return torch.from_numpy(np.ascontiguousarray(array))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    @override
    def __getitem__(self, indexes: object | Iterable[object]) -> Tensor:
        match indexes:
            case Iterable():
                tensors = map(self._getitem, indexes)  # pyright: ignore[reportUnknownArgumentType]

                return pad_sequence(list(tensors), dim=0, value=torch.nan)

            case _:
                return self._getitem(indexes)

    @override
    def __len__(self) -> int:
        raise NotImplementedError
