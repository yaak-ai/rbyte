from collections.abc import Callable

from optree import PyTree, tree_broadcast_map


class TreeBroadcastMapper:
    """A `pipefunc.PipeFunc`-friendly wrapper of `optree.tree_broadcast_map`."""

    __name__ = __qualname__

    def __call__(  # noqa: PLR0913
        self,
        *,
        func: Callable,
        left: PyTree,
        right: PyTree,
        is_leaf: Callable[..., bool] | None = None,
        none_is_leaf: bool = False,
        namespace: str = "",
    ) -> PyTree:
        return tree_broadcast_map(
            func,
            left,
            right,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
