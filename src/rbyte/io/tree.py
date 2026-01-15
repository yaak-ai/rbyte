from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from optree import PyTree, tree_broadcast_map

T = TypeVar("T")
U = TypeVar("U")


class TreeBroadcastMapper:
    """A `pipefunc.PipeFunc`-friendly wrapper of `optree.tree_broadcast_map`."""

    __name__ = __qualname__

    def __call__(  # noqa: PLR0913
        self,
        *,
        func: Callable[..., U],
        left: PyTree[T],
        right: PyTree[T],
        is_leaf: Callable[[T], bool] | None = None,
        none_is_leaf: bool = False,
        namespace: str = "",
    ) -> PyTree[U]:
        return tree_broadcast_map(
            func,
            left,
            right,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
