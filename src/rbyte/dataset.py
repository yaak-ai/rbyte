from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from enum import StrEnum, unique
from functools import cache
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, override

import more_itertools as mit
import polars as pl
import torch
from optree import tree_map, tree_structure, tree_transpose
from pipefunc import Pipeline
from pipefunc._pipeline._types import OUTPUT_TYPE
from pipefunc.map import load_outputs
from pydantic import (
    ConfigDict,
    InstanceOf,
    RootModel,
    StringConstraints,
    field_validator,
    model_validator,
    validate_call,
)
from structlog import get_logger
from tensordict import TensorDict
from torch.utils.data import Dataset as TorchDataset

from rbyte.batch import BATCH_KEYS_DEFAULT, Batch, BatchKeys, BatchMeta
from rbyte.config import BaseModel, HydraConfig
from rbyte.io.base import TensorSource

__all__ = ["Dataset"]

logger = get_logger(__name__)

type Id = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[\x00-\x7F]+$")
]


class SourceConfig(BaseModel):
    source: HydraConfig[TensorSource]  # pyright: ignore[reportMissingTypeArgument]
    index_column: str


class SourcesConfig(RootModel[dict[Id, dict[Id, SourceConfig]]]):
    @model_validator(mode="after")
    def _validate_index_column(self) -> Self:
        index_columns = {
            input_id: tuple(
                (source_id, source_cfg.index_column)
                for source_id, source_cfg in input_source_cfg.items()
            )
            for input_id, input_source_cfg in self.root.items()
        }

        match list(mit.unique(index_columns.values())):
            case []:
                pass

            case [input_index_columns]:
                if not mit.all_unique(
                    index_column for _, index_column in input_index_columns
                ):
                    msg = "`index_column` values not unique"
                    raise ValueError(msg)

            case _:
                msg = "`index_column` values not consistent"
                raise ValueError(msg)

        return self


class BasePipelineConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")
    inputs: Sequence[dict[str, Any]]
    run_folder: str | Path | None = None
    return_results: bool = True

    @field_validator("inputs", mode="after")
    @classmethod
    def _validate_inputs(
        cls, value: Sequence[dict[str, Any]]
    ) -> Sequence[dict[str, Any]]:
        if not mit.all_equal(map(tree_structure, value)):  # pyright: ignore[reportArgumentType]
            msg = "inputs have different structures"
            raise ValueError(msg)

        return value

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.return_results and self.run_folder is None:
            msg = "`run_folder` must be set when `return_results` is False"
            raise ValueError(msg)

        return self


class PipelineInstanceConfig(BasePipelineConfig):
    executor: InstanceOf[Executor] | dict[OUTPUT_TYPE, InstanceOf[Executor]] | None = (
        None
    )
    pipeline: InstanceOf[Pipeline]


class PipelineHydraConfig(BasePipelineConfig):
    executor: (
        HydraConfig[Executor] | dict[OUTPUT_TYPE, HydraConfig[Executor]] | None
    ) = None
    pipeline: HydraConfig[Pipeline]


@unique
class Column(StrEnum):
    input_id = "__input_id"
    sample_idx = "__sample_idx"
    source_idxs = "__source_idxs"
    source_id = "__source.id"
    source_config = "__source.config"
    source_index_column = "__source.index_column"


class _ALL_TYPE:  # noqa: N801
    pass


_ALL = _ALL_TYPE()


class Dataset(TorchDataset[Batch]):
    @validate_call
    def __init__(
        self,
        *,
        samples: PipelineInstanceConfig | PipelineHydraConfig,
        sources: SourcesConfig | None = None,
        enable_batched_sampling: bool = True,
    ) -> None:
        super().__init__()

        logger.debug("initializing dataset")

        self._samples: pl.DataFrame = self._build_samples(samples)
        logger.debug(
            "built samples",
            height=self._samples.height,
            size=f"{self._samples.estimated_size(unit := 'gb'):.3f} {unit}",
        )

        self._sources: pl.DataFrame | None = (
            self._build_sources(sources) if sources is not None else None
        )

        if enable_batched_sampling:
            self.__getitems__: Callable[[Sequence[int]], Batch] = self._getitems

    @property
    def samples(self) -> pl.DataFrame:
        return self._samples

    @property
    def sources(self) -> pl.DataFrame | None:
        return self._sources

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, validate_default=False)
    )
    def get_batch(
        self,
        index: int | Sequence[int] | slice | range,
        *,
        keys: BatchKeys = BATCH_KEYS_DEFAULT,
    ) -> Batch:
        subkeys: dict[Literal["data", "meta"], set[_ALL_TYPE | str]] = {
            "data": set(),
            "meta": set(),
        }
        for key in keys:
            match key:
                case "data" | "meta":
                    subkeys[key].add(_ALL)

                case ("data" | "meta", _):
                    subkeys[key[0]].add(key[1])

        for v in subkeys.values():
            if _ALL in v and len(v) > 1:
                v.remove(_ALL)

        samples = self.samples[index]
        batch_size = [samples.height]

        if subkeys_data := subkeys["data"]:
            if self.sources is not None:
                source_idx_cols = self.sources[Column.source_index_column].unique()
                sources = (
                    samples.lazy()
                    .join(self.sources.lazy(), on=Column.input_id, how="left")
                    .with_columns(
                        pl.coalesce(
                            pl.when(pl.col(Column.source_index_column) == idx_col).then(
                                idx_col
                            )
                            for idx_col in source_idx_cols
                        ).alias(Column.source_idxs)
                    )
                    .group_by(Column.source_id)
                    .agg(Column.source_config, Column.source_idxs)
                    .filter(
                        True
                        if _ALL in subkeys_data
                        else pl.col(Column.source_id).is_in(subkeys_data)
                    )
                )

                source_data = {
                    row[Column.source_id]: torch.stack([
                        self._get_source(source)[idxs]  # pyright: ignore[reportUnknownMemberType]
                        for (source, idxs) in zip(
                            row[Column.source_config],
                            row[Column.source_idxs],
                            strict=True,
                        )
                    ])
                    for row in sources.collect().iter_rows(named=True)
                }
            else:
                source_data = {}

            sample_data_cols = (
                pl.all()
                if _ALL in subkeys_data
                else pl.col(subkeys_data - source_data.keys())  # pyright: ignore[reportArgumentType]
            ).exclude(Column.sample_idx, Column.input_id)

            samples_subset = samples.select(sample_data_cols.to_physical())

            try:
                sample_data = samples_subset.to_torch(return_type="dict")
            except TypeError:
                sample_data = samples_subset.to_dict(as_series=False)

            data = TensorDict(source_data | sample_data, batch_size=batch_size)  # pyright: ignore[reportArgumentType]

        else:
            data = None

        if subkeys_meta := subkeys["meta"]:
            meta = BatchMeta(
                sample_idx=(
                    samples[Column.sample_idx].to_torch()
                    if _ALL in subkeys_meta or "sample_idx" in subkeys_meta
                    else None
                ),
                input_id=(
                    samples[Column.input_id].to_list()
                    if _ALL in subkeys_meta or "input_id" in subkeys_meta
                    else None
                ),
                batch_size=batch_size,
            )
        else:
            meta = None

        return Batch(data=data, meta=meta, batch_size=batch_size)

    def _getitems(self, index: Sequence[int]) -> Batch:
        return self.get_batch(index)

    @override
    def __getitem__(self, index: int) -> Batch:
        return self.get_batch(index)

    def __len__(self) -> int:
        return len(self.samples)

    @cache  # noqa: B019
    def _get_source(self, config: str) -> TensorSource:  # pyright: ignore[reportUnknownParameterType, reportMissingTypeArgument] # noqa: PLR6301
        return HydraConfig[TensorSource].model_validate_json(config).instantiate()  # pyright: ignore[reportUnknownVariableType, reportMissingTypeArgument]

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
                executor: Executor | dict[OUTPUT_TYPE, Executor] | None = tree_map(  # pyright: ignore[reportAssignmentType]
                    HydraConfig[Executor].instantiate,
                    samples.executor,  # pyright: ignore[reportArgumentType]
                )

        pipeline.print_documentation()
        inputs = tree_transpose(  # pyright: ignore[reportUnknownVariableType]
            tree_structure(list(range(len(samples.inputs)))),  # pyright: ignore[reportArgumentType]
            tree_structure(samples.inputs[0]),  # pyright: ignore[reportArgumentType]
            samples.inputs,  # pyright: ignore[reportArgumentType]
        )

        results = pipeline.map(
            inputs=inputs,  # pyright: ignore[reportArgumentType]
            executor=executor,
            **samples.model_dump(exclude={"pipeline", "inputs", "executor"}),
        )

        if pipeline.profile:
            pipeline.print_profiling_stats()

        output_name: str = pipeline.unique_leaf_node.output_name  # pyright: ignore[reportUnknownMemberType, reportAssignmentType]

        result: pl.DataFrame = (
            results[output_name].output
            if results
            else load_outputs(output_name, run_folder=samples.run_folder)  # pyright: ignore[reportArgumentType]
        )

        return (
            result.lazy()
            .cast({Column.input_id: pl.Enum(sorted(result[Column.input_id].unique()))})
            .sort(Column.input_id)
            .with_row_index(Column.sample_idx)
            .collect()
            .rechunk()
        )

    @classmethod
    def _build_sources(cls, sources: SourcesConfig) -> pl.DataFrame:
        logger.debug("building sources")

        input_id_enum = pl.Enum(sorted(sources.root.keys()))

        return (
            pl.DataFrame(
                [
                    {
                        Column.input_id: input_id,
                        (k := "__source"): [
                            source_cfg.model_dump(exclude={"source"})
                            | {
                                "id": source_id,
                                "config": source_cfg.source.model_dump_json(  # pyright: ignore[reportUnknownMemberType]
                                    by_alias=True
                                ),
                            }
                            for source_id, source_cfg in input_cfg.items()
                        ],
                    }
                    for input_id, input_cfg in sources.root.items()
                ],
                schema_overrides={Column.input_id: input_id_enum},
            )
            .explode(k)
            .unnest(k)
            .select(Column.input_id, pl.exclude(Column.input_id).name.prefix(f"{k}."))
            .rechunk()
        )
