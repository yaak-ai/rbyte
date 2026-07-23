from collections.abc import Callable, Sequence

from optree import GetItemEntry, PyTree, PyTreeAccessor, PyTreeKind, tree_broadcast_map


class TreeItemGetter:
    """A `pipefunc.PipeFunc`-friendly PyTree key-path lookup."""

    __name__ = __qualname__

    def __init__(self, *, key_path: Sequence[object]) -> None:
        self._accessor = PyTreeAccessor(
            GetItemEntry(key, object, PyTreeKind.CUSTOM) for key in key_path
        )

    def __call__(self, *, mapping: PyTree) -> object:
        return self._accessor(mapping)


class TreeBroadcastMapper:
    """A `pipefunc.PipeFunc`-friendly wrapper of `optree.tree_broadcast_map`."""

    __name__ = __qualname__

    def __call__(  # ruff:ignore[too-many-arguments]
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
