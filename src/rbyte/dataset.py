from collections.abc import Sequence
from concurrent.futures import Executor
from enum import StrEnum, unique
from functools import cache
from typing import Annotated, Any, ClassVar, Literal, override

import polars as pl
import torch
from optree import tree_map, tree_structure, tree_transpose
from pipefunc import Pipeline
from pipefunc._pipeline._types import OUTPUT_TYPE
from pydantic import ConfigDict, StringConstraints, validate_call
from structlog import get_logger
from tensordict import TensorDict
from torch.utils.data import Dataset as TorchDataset

from rbyte.batch import BATCH_KEYS_DEFAULT, Batch, BatchKeys, BatchMeta
from rbyte.config import BaseModel, HydraConfig
from rbyte.io.base import TensorSource
from rbyte.utils.tensor import pad_sequence

__all__ = ["Dataset"]

logger = get_logger(__name__)

type Id = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[\x00-\x7F]+$")
]


class SourceConfig(BaseModel):
    source: HydraConfig[TensorSource]  # pyright: ignore[reportMissingTypeArgument]
    index_column: str


type SourcesConfig = dict[Id, dict[Id, SourceConfig]]


class PipelineConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    pipeline: HydraConfig[Pipeline]
    inputs: dict[str, Any]
    executor: (
        HydraConfig[Executor] | dict[OUTPUT_TYPE, HydraConfig[Executor]] | None
    ) = None
    return_results: Literal[True] = True


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
        self, *, samples: PipelineConfig, sources: SourcesConfig | None = None
    ) -> None:
        super().__init__()

        logger.debug("initializing dataset")

        self._samples: pl.DataFrame = self._build_samples(samples)
        logger.debug("built samples", length=len(self._samples))

        self._sources: pl.DataFrame | None = (
            self._build_sources(sources) if sources is not None else None
        )

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
                    row[Column.source_id]: pad_sequence(
                        [
                            self._get_source(source)[idxs]  # pyright: ignore[reportUnknownMemberType]
                            for (source, idxs) in zip(
                                row[Column.source_config],
                                row[Column.source_idxs],
                                strict=True,
                            )
                        ],
                        dim=1,
                        value=torch.nan,
                    )
                    for row in sources.collect().iter_rows(named=True)
                }
            else:
                source_data = {}

            sample_data_cols = (
                pl.all()
                if _ALL in subkeys_data
                else pl.col(subkeys_data - source_data.keys())  # pyright: ignore[reportArgumentType]
            ).exclude(Column.sample_idx, Column.input_id)

            sample_data = samples.select(sample_data_cols.to_physical()).to_dict(
                as_series=False
            )

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

    def __getitems__(self, index: Sequence[int]) -> Batch:  # noqa: PLW3201
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
    def _build_samples(cls, samples: PipelineConfig) -> pl.DataFrame:
        logger.debug("building samples")
        pipeline = samples.pipeline.instantiate()
        pipeline.print_documentation()

        input_ids, input_values = zip(*samples.inputs.items(), strict=False)
        outer = tree_structure(list(range(len(input_values))))  # pyright: ignore[reportArgumentType]
        inner = tree_structure(input_values[0])
        inputs = tree_transpose(outer, inner, input_values)  # pyright: ignore[reportUnknownVariableType, reportArgumentType]

        executor: Executor | dict[OUTPUT_TYPE, Executor] = tree_map(  # pyright: ignore[reportAssignmentType]
            HydraConfig[Executor].instantiate,
            samples.executor,  # pyright: ignore[reportArgumentType]
        )

        results = pipeline.map(
            inputs=inputs,  # pyright: ignore[reportArgumentType]
            executor=executor,
            **samples.model_dump(exclude={"pipeline", "inputs", "executor"}),
        )
        output_name = pipeline.unique_leaf_node.output_name  # pyright: ignore[reportUnknownMemberType]
        output: Sequence[pl.DataFrame] = results[output_name].output  # pyright: ignore[reportArgumentType]

        input_id_enum = pl.Enum(input_ids)

        return (
            pl.concat(
                [
                    df.select(
                        pl.lit(input_id).cast(input_id_enum).alias(Column.input_id),
                        pl.col(sorted(df.collect_schema().names())),
                    )
                    for input_id, df in zip(input_ids, output, strict=True)
                ],
                how="vertical",
            )
            .sort(Column.input_id)
            .with_row_index(Column.sample_idx)
            .rechunk()
        )

    @classmethod
    def _build_sources(cls, sources: dict[Id, dict[Id, SourceConfig]]) -> pl.DataFrame:
        logger.debug("building sources")

        input_id_enum = pl.Enum(categories=sources.keys())

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
                    for input_id, input_cfg in sources.items()
                ],
                schema_overrides={Column.input_id: input_id_enum},
            )
            .explode(k)
            .unnest(k)
            .select(Column.input_id, pl.exclude(Column.input_id).name.prefix(f"{k}."))
            .rechunk()
        )
