from collections.abc import Callable, Iterable, Sequence, Sized
from typing import Any, Literal, Protocol, runtime_checkable

import torchdata.nodes as tn
from pydantic import InstanceOf, NonNegativeInt, PositiveInt, validate_call
from torch import Generator
from torch.utils.data import BatchSampler, RandomSampler, Sampler, SequentialSampler
from torchdata.nodes.loader import LoaderIterator


def collate_identity[T](x: T) -> T:
    return x


@runtime_checkable
class BatchIndexableDataset(Protocol):
    def __getitems__(self, index: Sequence[int]) -> object: ...  # ruff:ignore[bad-dunder-method-name]
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


class NodeDataLoader[T](Iterable[T], Sized):
    """https://meta-pytorch.org/data/main/migrate_to_nodes_from_utils.html"""

    @validate_call
    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        dataset: InstanceOf[BatchIndexableDataset],
        batch_size: PositiveInt = 1,
        shuffle: bool | None = None,
        sampler: InstanceOf[Sampler[int]] | None = None,
        batch_sampler: InstanceOf[Sampler[Sequence[int]]] | None = None,
        num_workers: NonNegativeInt = 1,
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
    ) -> None:
        self._dataset = dataset

        if sampler is not None and shuffle:
            msg = "sampler option is mutually exclusive with shuffle"
            raise ValueError(msg)

        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                msg = (
                    "batch_sampler option is mutually exclusive with "
                    "batch_size, shuffle, sampler, and drop_last"
                )
                raise ValueError(msg)
        else:
            if sampler is None:
                sampler = (
                    RandomSampler(dataset, generator=generator)
                    if shuffle
                    else SequentialSampler(dataset)
                )

            batch_sampler = BatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last
            )

        self._sampler = batch_sampler

        node = tn.SamplerWrapper(self._sampler)
        map_fn = MapAndCollate(
            dataset, collate_fn if collate_fn is not None else collate_identity
        )
        node = (
            tn.ParallelMapper(
                source=node,
                map_fn=map_fn,
                num_workers=num_workers,
                in_order=in_order,
                method=method,
                multiprocessing_context=multiprocessing_context,
                max_concurrent=max_concurrent,
                snapshot_frequency=snapshot_frequency,
                prebatch=prebatch,
            )
            if num_workers
            else tn.Mapper(source=node, map_fn=map_fn)
        )

        if pin_memory:
            node = tn.PinMemory(node, pin_memory_device=pin_memory_device)

        if num_workers:
            node = tn.Prefetcher(node, prefetch_factor=num_workers * prefetch_factor)

        self._loader = tn.Loader(node)

    def __iter__(self) -> LoaderIterator[T]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._sampler)  # ty:ignore[invalid-argument-type]

    def state_dict(self) -> dict[str, Any]:
        return self._loader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._loader.load_state_dict(state_dict)

    @property
    def dataset(self) -> BatchIndexableDataset:
        return self._dataset
