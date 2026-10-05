"""A `StreamSource` that applies a module to every decoded batch of frames."""

from collections.abc import Callable, Sequence
from typing import Any, final, override

from torch import Tensor

from rbyte.streams.base import StreamSource


@final
class TransformedSource(StreamSource[Any]):
    """Wrap `source` and apply `transform` to what it returns.

    Unlike `TorchCodecVideoSource(transforms=...)`, which only accepts decoder-side
    transforms (swscale), this runs an arbitrary torch callable AFTER decoding.
    It exists so the training data path can call the exact function serving
    calls -- e.g. nero's native-RGB -> model-grid preprocessing (rmind
    `rmind.data.nero_image.NeroImagePreprocess`), where a decoder-side resize
    would be a different kernel from the one the robot runs.
    """

    def __init__(
        self, *, source: StreamSource[Any], transform: Callable[[Tensor], Tensor]
    ) -> None:
        super().__init__()
        self._source = source
        self._transform = transform

    @override
    def __getitem__(self, indexes: Any | Sequence[Any]) -> Tensor:
        return self._transform(self._source[indexes])
