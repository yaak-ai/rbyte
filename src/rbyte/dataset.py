from collections.abc import Sequence
from concurrent.futures import Executor
from enum import StrEnum, auto, unique
from io import BytesIO
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import local
from typing import TYPE_CHECKING, Any, Self, override

import more_itertools as mit
import polars as pl
from optree import tree_map
from pipefunc.map import load_outputs
from pydantic import DirectoryPath, InstanceOf, TypeAdapter, validate_call
from structlog import get_logger
from tensordict import NonTensorStack, TensorDict
from torch.utils.data import Dataset as TorchDataset

from rbyte.config import (
    HydraConfig,
    PipelineHydraConfig,
    PipelineInstanceConfig,
    StreamsConfig,
)
from rbyte.streams.base import StreamSource
from rbyte.types import Batch, BatchMeta

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["Dataset"]

logger = get_logger(__name__)


class _StreamSourceThreadCache(local):
    sources: dict[tuple[str, str], StreamSource]


@unique
class MetaColumn(StrEnum):
    input_id = auto()


class Dataset(TorchDataset[Batch]):  # ruff:ignore[eq-without-hash]
    __slots__ = ("_data", "_meta", "_stream_source_cache", "_streams")

    @staticmethod
    def _validate_meta(meta: pl.DataFrame) -> None:
        input_id = MetaColumn.input_id
        if input_id not in meta.schema:
            msg = "`meta` must contain an `input_id` column"
            raise ValueError(msg)
        if meta.get_column(input_id).has_nulls():
            msg = "`meta.input_id` must not contain nulls"
            raise ValueError(msg)
        if (dtype := meta.schema[input_id]).base_type() not in {pl.String, pl.Enum}:
            msg = f"`meta.input_id` must have String or Enum dtype, got {dtype}"
            raise ValueError(msg)

    @validate_call
    def __init__(
        self,
        *,
        data: InstanceOf[TensorDict],
        meta: InstanceOf[pl.DataFrame],
        streams: StreamsConfig | None,
    ) -> None:
        super().__init__()
        self._validate_meta(meta)
        data.auto_batch_size_(1)

        if (data_length := len(data)) != (meta_length := len(meta)):
            msg = f"`data` and `meta` lengths differ: {data_length} != {meta_length}"
            logger.error(msg, data_length=data_length, meta_length=meta_length)

            raise ValueError(msg)

        if streams is not None and (
            missing_stream_indexes := (
                {stream_config.index for stream_config in streams.values()}
                - (data_keys := set(data.keys(include_nested=True, leaves_only=True)))
            )
        ):
            logger.error(
                msg := "`data` missing stream indexes",
                data_keys=sorted(data_keys, key=str),
                indexes=sorted(missing_stream_indexes, key=str),
            )

            raise ValueError(msg)

        self._data = data.share_memory_().lock_()
        self._meta = meta
        self._streams = streams

        if self._streams is not None:
            self._stream_source_cache = _StreamSourceThreadCache()

    @classmethod
    @validate_call
    def from_config(
        cls,
        *,
        samples: PipelineInstanceConfig | PipelineHydraConfig,
        streams: StreamsConfig | None = None,
    ) -> Self:
        sample_df = cls._build_samples(samples)
        cls._validate_meta(sample_df)

        data = TensorDict(
            sample_df.select(pl.exclude(MetaColumn.input_id).to_physical()).to_torch(
                return_type="dict"
            )  # ty:ignore[invalid-argument-type]
        )

        meta = sample_df.select(MetaColumn.input_id).rechunk()

        return cls(data=data, meta=meta, streams=streams)

    @property
    def data(self) -> TensorDict:
        return self._data

    @property
    def meta(self) -> pl.DataFrame:
        return self._meta

    @property
    def streams(self) -> StreamsConfig | None:
        return self._streams

    @override
    def __getitem__(self, index: int) -> Batch:
        return self.get_batch([index])[0]  # ty: ignore[invalid-return-type]

    def __getitems__(self, index: Sequence[int]) -> Batch:  # ruff:ignore[bad-dunder-method-name]
        return self.get_batch(index)

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self,
        index: Sequence[int] | InstanceOf[range] | InstanceOf[slice],
        *,
        include_streams: bool | None = None,
        include_meta: bool = True,
    ) -> Batch:
        batch_data: TensorDict = self.data[index]  # ty: ignore[invalid-argument-type, invalid-assignment]
        meta: pl.DataFrame = self.meta[index]

        match include_streams, self.streams:
            case None | True, dict():
                stream_data = {}

                for stream_id, stream_config in self.streams.items():
                    stream_indexes = list(
                        zip(
                            meta["input_id"],
                            batch_data[stream_config.index].tolist(),
                            strict=True,
                        )
                    )

                    grouped_stream_indexes = mit.map_reduce(
                        stream_indexes, keyfunc=itemgetter(0), valuefunc=itemgetter(1)
                    )

                    stream_items: dict[str, dict[int, Tensor]] = {}
                    for input_id, group_indexes in grouped_stream_indexes.items():
                        unique_indexes = sorted(set(mit.collapse(group_indexes)))
                        source = self._get_source(stream_id, input_id)
                        source_items = source[unique_indexes]

                        stream_items[input_id] = dict(
                            zip(unique_indexes, source_items, strict=True)
                        )

                    stream_batches = []
                    for input_id, input_stream_index in stream_indexes:
                        input_stream_items = stream_items[input_id]
                        stream_batches.append(
                            [input_stream_items[i] for i in input_stream_index]
                            if isinstance(input_stream_index, Sequence)
                            else input_stream_items[input_stream_index]
                        )

                    stream_data[stream_id] = stream_batches

                if batch_data.is_locked:
                    batch_data = batch_data.clone(recurse=True)

                batch_data = batch_data.update(stream_data, inplace=False)

            case True, None:
                msg = "`include_streams` is True but no streams specified"
                raise ValueError(msg)

            case _:
                pass

        batch_meta = (
            BatchMeta.from_dict({
                k: NonTensorStack(*v) for k, v in meta.to_dict().items()
            })
            if include_meta
            else None
        )

        return Batch(data=batch_data, meta=batch_meta).auto_batch_size_(1)

    def _get_source(self, stream_id: str, input_id: str) -> StreamSource:
        streams = self.streams
        if streams is None:
            msg = "streams not specified"
            raise RuntimeError(msg)

        key = (stream_id, input_id)
        cache = self._get_stream_source_cache()

        try:
            return cache[key]
        except KeyError:
            source = streams[stream_id].sources[input_id].instantiate()
            cache[key] = source

            return source

    def _get_stream_source_cache(self) -> dict[tuple[str, str], StreamSource]:
        try:
            return self._stream_source_cache.sources
        except AttributeError:
            self._stream_source_cache.sources = {}

            return self._stream_source_cache.sources

    @classmethod
    def _build_samples(
        cls, samples: PipelineInstanceConfig | PipelineHydraConfig
    ) -> pl.DataFrame:
        logger.debug("building samples")

        match samples:
            case PipelineInstanceConfig():
                pipeline = samples.pipeline
                executor = samples.executor

            case PipelineHydraConfig():
                pipeline = samples.pipeline.instantiate()
                executor: Executor | dict[str | tuple[str, ...], Executor] | None = (  # ty: ignore[invalid-assignment]
                    tree_map(
                        HydraConfig[Executor].instantiate,
                        samples.executor,  # ty: ignore[invalid-argument-type]
                    )
                )

        output_name = pipeline.unique_leaf_node.output_name
        results = pipeline.map(
            executor=executor, **samples.model_dump(exclude={"pipeline", "executor"})
        )

        if pipeline.profile:
            logger.debug("pipeline profiling stats:")
            pipeline.print_profiling_stats()

        return (
            results[output_name].output  # ty:ignore[invalid-argument-type]
            if results
            else load_outputs(output_name, run_folder=samples.run_folder)  # ty: ignore[invalid-argument-type]
        )

    @validate_call
    def save(self, path: DirectoryPath) -> None:
        logger.debug("saving dataset", dataset=self, path=path.resolve().as_posix())

        with TemporaryDirectory(dir=path.parent) as txn_dir:
            txn_path = Path(txn_dir)
            staged_path = txn_path / "staged"
            previous_path = txn_path / "previous"
            staged_path.mkdir()

            self._data.memmap(
                staged_path / "data", copy_existing=True, existsok=True, robust_key=True
            )
            self._meta.write_parquet(staged_path / "meta.parquet")

            if self._streams is not None:
                streams_json = TypeAdapter(StreamsConfig).dump_json(self._streams)
                with (staged_path / "streams.json").open("wb") as f:
                    f.write(streams_json)

            path.rename(previous_path)
            try:
                staged_path.rename(path)
            except BaseException:
                previous_path.rename(path)
                raise

    @classmethod
    @validate_call
    def load(cls, path: DirectoryPath) -> Self:
        logger.debug("loading dataset", path=path.resolve().as_posix())
        data = TensorDict.load_memmap(path / "data", robust_key=True)
        meta = pl.read_parquet(path / "meta.parquet")

        try:
            with (path / "streams.json").open() as f:
                streams = TypeAdapter(StreamsConfig).validate_json(f.read())
        except FileNotFoundError:
            streams = None

        return cls(data=data, meta=meta, streams=streams)

    def __getstate__(self) -> dict[str, Any]:
        data = self._data

        meta = BytesIO()
        self._meta.write_parquet(meta)

        streams = (
            TypeAdapter(StreamsConfig).dump_json(self._streams)
            if self._streams is not None
            else None
        )

        return {"data": data, "meta": meta, "streams": streams}

    def __setstate__(self, state: dict[str, Any]) -> None:
        state["meta"] = pl.read_parquet(state["meta"])

        if (v := state[k := "streams"]) is not None:
            state[k] = TypeAdapter(StreamsConfig).validate_json(v)

        self.__init__(**state)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dataset):
            return NotImplemented

        return all((
            (self.data == other.data).all(),
            self.meta.equals(other.meta),
            self.streams == other.streams,
        ))
