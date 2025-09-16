from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import final, override

import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from pydantic import validate_call
from torch import Tensor

from rbyte.types import TensorSource


@final
class NumpyTensorSource(TensorSource[object]):
    @validate_call
    def __init__(
        self,
        path: PathLike[str],
        select: Sequence[str] | None = None,
        index_transform: Callable[..., object] | None = None,
    ) -> None:
        super().__init__()

        self._path = Path(path)
        self._select = select or ...
        self._index_transform = index_transform

    @cached_property
    def _path_posix(self) -> str:
        return self._path.resolve().as_posix()

    def _getitem(self, index: object) -> Tensor:
        if self._index_transform is not None:
            index = self._index_transform(index)

        path = self._path_posix.format(index)
        array = structured_to_unstructured(np.load(path)[self._select])

        return torch.from_numpy(np.ascontiguousarray(array))

    @override
    def __getitem__(self, indexes: object | Iterable[object]) -> Tensor:
        match indexes:
            case Iterable():
                return torch.stack([self._getitem(i) for i in indexes])

            case _:
                return self._getitem(indexes)

    @override
    def __len__(self) -> int:
        raise NotImplementedError
