from collections.abc import Mapping, Sequence
from enum import StrEnum, unique
from functools import cache
from typing import Annotated

import polars as pl
import torch
from pydantic import Field, StringConstraints, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tensordict import TensorDict
from torch.utils.data import Dataset as TorchDataset

from rbyte.batch import Batch, BatchMeta
from rbyte.config import BaseModel, HydraConfig
from rbyte.io.base import TensorSource
from rbyte.io.table.base import TableBuilder
from rbyte.sample.base import SampleBuilder
from rbyte.utils.functional import pad_sequence

__all__ = ["Dataset"]

logger = get_logger(__name__)

type Id = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[\x00-\x7F]+$")
]


class SourceConfig(BaseModel):
    source: HydraConfig[TensorSource]
    index_column: str


class InputConfig(BaseModel):
    sources: Mapping[Id, SourceConfig] = Field(min_length=1)
    table_builder: HydraConfig[TableBuilder]


@unique
class Column(StrEnum):
    input_id = "__input_id"
    sample_idx = "__sample_idx"
    source_idxs = "__source_idxs"
    source_id = "__source.id"
    source_config = "__source.config"
    source_index_column = "__source.index_column"


class Dataset(TorchDataset[TensorDict]):
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self,
        inputs: Annotated[Mapping[Id, InputConfig], Field(min_length=1)],
        sample_builder: HydraConfig[SampleBuilder],
    ) -> None:
        logger.debug("initializing dataset")

        super().__init__()

        _sample_builder = sample_builder.instantiate()
        samples: Mapping[str, pl.DataFrame] = {}
        for input_id, input_cfg in inputs.items():
            with bound_contextvars(input_id=input_id):
                table = input_cfg.table_builder.instantiate().build()
                samples[input_id] = _sample_builder.build(table)
                logger.debug(
                    "built samples",
                    rows=table.select(pl.len()).item(),
                    samples=samples[input_id].select(pl.len()).item(),
                )

        input_id_enum = pl.Enum(sorted(samples))

        self._samples: pl.DataFrame = (
            pl.concat(
                [
                    df.select(
                        pl.lit(input_id).cast(input_id_enum).alias(Column.input_id),
                        pl.col(sorted(df.collect_schema().names())),
                    )
                    for input_id, df in samples.items()
                ],
                how="vertical",
            )
            .sort(Column.input_id)
            .with_row_index(Column.sample_idx)
            .rechunk()
        )

        self._sources: pl.DataFrame = (
            pl.DataFrame(
                [
                    {
                        Column.input_id: input_id,
                        (k := "__source"): [
                            source_cfg.model_dump(exclude={"source"})
                            | {
                                "id": source_id,
                                "config": source_cfg.source.model_dump_json(
                                    by_alias=True
                                ),
                            }
                            for source_id, source_cfg in input_cfg.sources.items()
                        ],
                    }
                    for input_id, input_cfg in inputs.items()
                ],
                schema_overrides={Column.input_id: input_id_enum},
            )
            .explode(k)
            .unnest(k)
            .select(Column.input_id, pl.exclude(Column.input_id).name.prefix(f"{k}."))
            .rechunk()
        )

    @property
    def samples(self) -> pl.DataFrame:
        return self._samples

    @property
    def sources(self) -> pl.DataFrame:
        return self._sources

    @cache  # noqa: B019
    def _get_source(self, config: str) -> TensorSource:  # noqa: PLR6301
        return HydraConfig[TensorSource].model_validate_json(config).instantiate()

    def __getitems__(self, indexes: Sequence[int]) -> Batch:  # noqa: PLW3201
        samples = self.samples[indexes]
        batch_size = [samples.height]

        source_idx_cols = self._sources[Column.source_index_column].unique()

        sources = (
            samples.lazy()
            .join(self.sources.lazy(), on=Column.input_id, how="left")
            .with_columns(
                pl.coalesce(
                    pl.when(pl.col(Column.source_index_column) == idx_col).then(idx_col)
                    for idx_col in source_idx_cols
                ).alias(Column.source_idxs)
            )
            .group_by(Column.source_id)
            .agg(Column.source_config, Column.source_idxs)
        )

        tensors: Mapping[str, torch.Tensor] = {
            row[Column.source_id]: pad_sequence(
                [
                    self._get_source(source)[idxs]
                    for (source, idxs) in zip(
                        row[Column.source_config], row[Column.source_idxs], strict=True
                    )
                ],
                dim=1,
                value=torch.nan,
            )
            for row in sources.collect().iter_rows(named=True)
        }

        table: Mapping[str, Sequence[object]] = samples.select(
            pl.exclude(Column.sample_idx, Column.input_id).to_physical()
        ).to_dict(as_series=False)

        data = TensorDict(tensors | table, batch_size=batch_size)  # pyright: ignore[reportArgumentType]

        meta = BatchMeta(
            sample_idx=samples[Column.sample_idx].to_torch(),  # pyright: ignore[reportCallIssue]
            input_id=samples[Column.input_id].to_list(),  # pyright: ignore[reportCallIssue]
            batch_size=batch_size,  # pyright: ignore[reportCallIssue]
        )

        return Batch(data=data, meta=meta, batch_size=batch_size)  # pyright: ignore[reportCallIssue]

    def __len__(self) -> int:
        return len(self.samples)
