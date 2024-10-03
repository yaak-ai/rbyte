from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from jaxtyping import Shaped
from torch import Tensor


@runtime_checkable
class FrameReader(Protocol):
    def read(
        self, indexes: Iterable[int]
    ) -> Shaped[Tensor, "b h w c"] | Shaped[Tensor, "b c h w"]: ...
    def get_available_indexes(self) -> Sequence[int]: ...
