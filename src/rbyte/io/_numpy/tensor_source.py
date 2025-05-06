from collections.abc import Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import final, override

import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from pydantic import validate_call
from torch import Tensor

from rbyte.config.base import BaseModel
from rbyte.io.base import TensorSource


@final
class NumpyTensorSource(TensorSource[object]):
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
        array = structured_to_unstructured(np.load(path)[self._select])
        return torch.from_numpy(np.ascontiguousarray(array))  # pyright: ignore[reportUnknownMemberType]

    @override
    def __getitem__(self, indexes: object | Sequence[object]) -> Tensor:
        match indexes:
            case Sequence():
                tensors = map(self._getitem, indexes)  # pyright: ignore[reportUnknownArgumentType]

                return torch.stack(list(tensors))

            case _:
                return self._getitem(indexes)

    @override
    def __len__(self) -> int:
        raise NotImplementedError
