from collections.abc import Callable, Iterable, Sequence, Sized
from typing import Any, Literal, Protocol, runtime_checkable

import torch.distributed
import torchdata.nodes as tn
from pydantic import InstanceOf, PositiveInt, validate_call
from torch import Generator
from torch.utils.data import (
    BatchSampler,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
    default_collate,
)
from torchdata.nodes.loader import LoaderIterator


def collate_identity[T](x: T) -> T:
    return x


@runtime_checkable
class BatchIndexableDataset(Protocol):
    def __getitems__(self, index: Sequence[int]) -> object: ...  # noqa: PLW3201
    def __len__(self) -> int: ...


class MapAndCollate[T]:
    @validate_call
    def __init__(
        self, dataset: InstanceOf[BatchIndexableDataset], collate_fn: Callable[..., Any]
    ) -> None:
        self._dataset = dataset
        self._collate_fn = collate_fn

    def __call__(self, index: Sequence[int]) -> object:
        batch = self._dataset.__getitems__(index)
        return self._collate_fn(batch)


class TorchDataNodeDataLoader[T](Iterable[T], Sized):
    """https://meta-pytorch.org/data/main/migrate_to_nodes_from_utils.html"""

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        dataset: InstanceOf[BatchIndexableDataset],
        batch_size: int = 1,
        shuffle: bool | None = None,
        num_workers: PositiveInt = 1,
        collate_fn: Callable[..., Any] | None = None,
        pin_memory: bool = False,
        pin_memory_device: str = "",
        drop_last: bool = False,
        in_order: bool = True,
        method: Literal["thread", "process"] = "thread",
        multiprocessing_context: Literal["spawn", "forkserver", "fork"] | None = None,
        generator: InstanceOf[Generator] | None = None,
        prefetch_factor: int = 2,
        max_concurrent: int | None = None,
        snapshot_frequency: int = 1,
        prebatch: int | None = None,
        seed: int = 0,
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._generator = generator
        self._seed = seed
        self._method: Literal["thread", "process"] = method
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._pin_memory_device = pin_memory_device
        self._prefetch_factor = prefetch_factor
        self._node_kwargs = {
            "num_workers": num_workers,
            "in_order": in_order,
            "multiprocessing_context": multiprocessing_context,
            "max_concurrent": max_concurrent,
            "snapshot_frequency": snapshot_frequency,
            "prebatch": prebatch,
        }

        self._distributed_sampler: DistributedSampler[object] | None = None
        self._sampler: BatchSampler | None = None
        self._loader: tn.Loader[T] | None = None
        self._built_distributed: bool | None = None

    @staticmethod
    def _distributed_now() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _build(self) -> None:
        """Build the sampler and node graph.

        Deferred past ``__init__``: framework integrations (e.g. hydra +
        pytorch-lightning) construct the loader before ``trainer.fit``
        initializes the distributed process group, so the sharding decision
        must be taken at first iteration. Without a ``DistributedSampler``,
        every rank draws an identical (same-seed) sample sequence and the
        all-reduced gradient degenerates to a single rank's.

        Raises:
            ValueError: if ``method='process'`` is configured under
                distributed training (workers fork after CUDA init and
                deadlock).
        """
        distributed = self._distributed_now()
        self._built_distributed = distributed

        if distributed:
            if self._method != "thread":
                msg = (
                    "method='process' deadlocks under distributed training "
                    "(fork after CUDA init); use method='thread'"
                )
                raise ValueError(msg)
            self._distributed_sampler = DistributedSampler(
                self._dataset,  # pyright: ignore[reportArgumentType]
                shuffle=bool(self._shuffle),
                drop_last=self._drop_last,
                seed=self._seed,
            )
            sampler = self._distributed_sampler
        else:
            self._distributed_sampler = None
            sampler = (  # pyright: ignore[reportAssignmentType]
                RandomSampler(self._dataset, generator=self._generator)  # pyright: ignore[reportArgumentType]
                if self._shuffle
                else SequentialSampler(self._dataset)  # pyright: ignore[reportArgumentType]
            )

        self._sampler = BatchSampler(
            sampler, batch_size=self._batch_size, drop_last=self._drop_last
        )

        node = tn.SamplerWrapper(self._sampler)
        node = tn.ParallelMapper(
            source=node,
            map_fn=MapAndCollate(self._dataset, self._collate_fn or default_collate),
            method=self._method,
            **self._node_kwargs,  # pyright: ignore[reportArgumentType]
        )

        if self._pin_memory:
            node = tn.PinMemory(node, pin_memory_device=self._pin_memory_device)

        node = tn.Prefetcher(
            node,
            prefetch_factor=self._node_kwargs["num_workers"] * self._prefetch_factor,  # pyright: ignore[reportOperatorIssue]
        )

        self._loader = tn.Loader(node)

    def _ensure_built(self) -> None:
        # rebuild if the process group appeared after a premature build —
        # an unsharded graph under distributed training silently degenerates
        # the effective batch to a single rank's
        if (
            self._loader is not None
            and self._built_distributed is False
            and self._distributed_now()
        ):
            self._loader = None
        if self._loader is None:
            self._build()

    def set_epoch(self, epoch: int) -> None:
        """Forward the epoch to the DistributedSampler for per-epoch reshuffling."""
        if self._distributed_sampler is not None:
            self._distributed_sampler.set_epoch(epoch)

    def __iter__(self) -> LoaderIterator[T]:
        self._ensure_built()
        assert self._loader is not None  # noqa: S101
        return iter(self._loader)

    def __len__(self) -> int:
        self._ensure_built()
        assert self._sampler is not None  # noqa: S101
        return len(self._sampler)

    @property
    def dataset(self) -> BatchIndexableDataset:
        return self._dataset
