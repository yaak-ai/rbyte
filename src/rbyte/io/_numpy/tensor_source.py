from collections.abc import Iterable, Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import numpy as np
import torch
from numpy.lib.recfunctions import (
    structured_to_unstructured,  # pyright: ignore[reportUnknownVariableType]
)
from pydantic import validate_call
from torch import Tensor

from rbyte.config.base import BaseModel
from rbyte.io.base import TensorSource
from rbyte.utils.functional import pad_sequence

if TYPE_CHECKING:
    from types import EllipsisType


class NumpyTensorSource(TensorSource):
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self, path: PathLike[str], select: Sequence[str] | None = None
    ) -> None:
        super().__init__()

        self._path: Path = Path(path)
        self._select: Sequence[str] | EllipsisType = select or ...

    @cached_property
    def _path_posix(self) -> str:
        return self._path.resolve().as_posix()

    @override
    def __getitem__(self, indexes: Iterable[Any]) -> Tensor:
        tensors: list[Tensor] = []
        for index in indexes:
            path = self._path_posix.format(index)
            array = structured_to_unstructured(np.load(path)[self._select])  # pyright: ignore[reportUnknownVariableType]
            tensor = torch.from_numpy(np.ascontiguousarray(array))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            tensors.append(tensor)

        return pad_sequence(list(tensors), dim=0, value=torch.nan)

    @override
    def __len__(self) -> int:
        raise NotImplementedError
