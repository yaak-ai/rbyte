from collections.abc import Callable, Iterable, Sequence, Sized
from typing import Any, Literal, Protocol, runtime_checkable

import torchdata.nodes as tn
from pydantic import InstanceOf, validate_call
from pydantic.types import PositiveInt
from torch import Generator
from torch.utils.data import (
    BatchSampler,
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
        self,
        dataset: InstanceOf[BatchIndexableDataset],
        collate_fn: Callable[[list[T]], Any],
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
        collate_fn: Callable[[...], Any] | None = None,
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
    ) -> None:
        self._dataset = dataset

        sampler = (
            RandomSampler(dataset, generator=generator)
            if shuffle
            else SequentialSampler(dataset)
        )

        self._sampler = BatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last
        )

        node = tn.SamplerWrapper(self._sampler)
        node = tn.ParallelMapper(
            source=node,
            map_fn=MapAndCollate(dataset, collate_fn or default_collate),
            num_workers=num_workers,
            in_order=in_order,
            method=method,
            multiprocessing_context=multiprocessing_context,
            max_concurrent=max_concurrent,
            snapshot_frequency=snapshot_frequency,
            prebatch=prebatch,
        )

        if pin_memory:
            node = tn.PinMemory(node, pin_memory_device=pin_memory_device)

        node = tn.Prefetcher(node, prefetch_factor=num_workers * prefetch_factor)

        self._loader = tn.Loader(node)

    def __iter__(self) -> LoaderIterator[T]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._sampler)

    @property
    def dataset(self) -> BatchIndexableDataset:
        return self._dataset
