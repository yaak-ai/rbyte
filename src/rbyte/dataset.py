import math
from collections.abc import Sequence
from concurrent.futures import Executor
from enum import StrEnum, auto, unique
from io import BytesIO
from typing import TYPE_CHECKING, Annotated, Any, Self, override

import checkedframe as cf
import polars as pl
import torch
from cachetools import Cache, cachedmethod
from optree import tree_map
from pipefunc.map import load_outputs
from pydantic import DirectoryPath, InstanceOf, validate_call
from pydantic.functional_validators import AfterValidator
from pydantic.type_adapter import TypeAdapter
from structlog import get_logger
from tensordict import NonTensorStack, TensorDict
from torch.utils.data import Dataset as TorchDataset

from rbyte.config import (
    HydraConfig,
    PipelineHydraConfig,
    PipelineInstanceConfig,
    StreamsConfig,
)
from rbyte.types import Batch, BatchMeta, TensorSource

if TYPE_CHECKING:
    from pipefunc._pipeline._types import OUTPUT_TYPE

__all__ = ["Dataset"]

logger = get_logger(__name__)


@unique
class MetaColumn(StrEnum):
    input_id = auto()


class MetaSchema(cf.Schema):
    input_id = cf.Union(cf.String(), cf.Enum())


if not set(MetaColumn).issubset(MetaSchema.columns()):
    raise ValueError


class Dataset(TorchDataset[Batch]):  # noqa: PLW1641
    __slots__ = ("_data", "_meta", "_stream_source_cache", "_streams")

    @validate_call
    def __init__(
        self,
        *,
        data: InstanceOf[TensorDict],
        meta: Annotated[InstanceOf[pl.DataFrame], AfterValidator(MetaSchema.validate)],
        streams: StreamsConfig | None,
    ) -> None:
        super().__init__()
        if streams is not None and (
            missing_stream_indexes := (
                {stream_config.index for stream_config in streams.values()}
                - (data_keys := data.keys())
            )
        ):
            logger.error(
                msg := "`data` missing stream indexes",
                data_keys=sorted(data_keys),
                indexes=sorted(missing_stream_indexes),
            )

            raise ValueError(msg)

        self._data = data.auto_batch_size_(1).share_memory_().lock_()
        self._meta = meta
        self._streams = streams

        if self._streams is not None:
            self._stream_source_cache = Cache(maxsize=math.inf)

    @classmethod
    @validate_call
    def from_config(
        cls,
        *,
        samples: PipelineInstanceConfig | PipelineHydraConfig,
        streams: StreamsConfig | None = None,
    ) -> Self:
        samples = cls._build_samples(samples)
        samples = MetaSchema.validate(samples)

        data = TensorDict(
            samples.select(pl.exclude(MetaSchema.columns()).to_physical()).to_torch(
                return_type="dict"
            )
        )

        meta = samples.select(MetaSchema.columns()).rechunk()

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
        return self.get_batch([index])[0]

    def __getitems__(self, index: Sequence[int]) -> Batch:  # noqa: PLW3201
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
        data = self.data[index]
        meta = self.meta[index]

        match include_streams, self.streams:
            case None | True, dict():
                stream_data = {stream_id: [] for stream_id in self.streams}

                for sample, input_id in zip(data, meta["input_id"], strict=True):
                    for stream_id, stream_config in self.streams.items():
                        stream_index = sample[stream_config.index].tolist()
                        source = self._get_source(stream_id, input_id)
                        stream_data[stream_id].append(source[stream_index])

                stream_data = {k: torch.stack(v) for k, v in stream_data.items()}

                if data.is_locked:
                    data = data.clone(recurse=True)

                data = data.update(stream_data, inplace=False)

            case True, None:
                msg = "`include_streams` is True but no streams specified"
                raise ValueError(msg)

            case _:
                pass

        meta = (
            BatchMeta.from_dict({
                k: NonTensorStack(*v) for k, v in meta.to_dict().items()
            })
            if include_meta
            else None
        )

        return Batch(data=data, meta=meta).auto_batch_size_(1)

    @cachedmethod(lambda self: self._stream_source_cache)
    def _get_source(self, stream_id: str, input_id: str) -> TensorSource:
        return self.streams[stream_id].sources[input_id].instantiate()

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
                executor: Executor | dict[OUTPUT_TYPE, Executor] | None = tree_map(  # ty: ignore[invalid-assignment]
                    HydraConfig[Executor].instantiate,
                    samples.executor,  # ty: ignore[invalid-argument-type]
                )

        output_name = pipeline.unique_leaf_node.output_name
        results = pipeline.map(  # ty: ignore[missing-argument]
            executor=executor, **samples.model_dump(exclude={"pipeline", "executor"})
        )

        return (
            results[output_name].output
            if results
            else load_outputs(output_name, run_folder=samples.run_folder)  # ty: ignore[invalid-argument-type]
        )

    @validate_call
    def save(self, path: DirectoryPath) -> None:
        logger.debug("saving dataset", dataset=self, path=path.resolve().as_posix())

        self._data.memmap(
            path / "data", copy_existing=True, existsok=True, robust_key=True
        )
        self._meta.write_parquet(path / "meta.parquet")
        streams_json = TypeAdapter(StreamsConfig).dump_json(self._streams)
        with (path / "streams.json").open("wb") as f:
            f.write(streams_json)

    @classmethod
    @validate_call
    def load(cls, path: DirectoryPath) -> None:
        logger.debug("loading dataset", path=path.resolve().as_posix())
        data = TensorDict.load_memmap(path / "data", robust_key=True)
        meta = pl.read_parquet(path / "meta.parquet")

        with (path / "streams.json").open() as f:
            streams = TypeAdapter(StreamsConfig).validate_json(f.read())

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
