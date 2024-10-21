from collections.abc import Mapping, Sequence
from enum import StrEnum, unique
from functools import cache
from typing import Annotated

import more_itertools as mit
import polars as pl
import torch
from pydantic import ConfigDict, Field, StringConstraints, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tensordict import TensorDict
from torch.utils.data import Dataset as TorchDataset

from rbyte.batch import Batch, BatchMeta
from rbyte.config.base import BaseModel, HydraConfig
from rbyte.io.frame.base import FrameReader
from rbyte.io.table.base import TableBuilderBase
from rbyte.sample.base import SampleTableBuilder

__all__ = ["Dataset"]

logger = get_logger(__name__)

type Id = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[\x00-\x7F]+$")
]


class FrameSourceConfig(BaseModel):
    reader: HydraConfig[FrameReader]
    index_column: str


class TableSourceConfig(BaseModel):
    builder: HydraConfig[TableBuilderBase]


class SourcesConfig(BaseModel):
    frame: Mapping[Id, FrameSourceConfig] = Field(min_length=1)
    table: TableSourceConfig | None = None


@unique
class Column(StrEnum):
    input_id = "__input_id"
    sample_idx = "__sample_idx"
    frame_idx = "__frame_idx"
    source_id = "source.id"
    source_reader = "source.reader"
    source_index_column = "source.index_column"


class Dataset(TorchDataset[TensorDict]):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        inputs: Annotated[Mapping[Id, SourcesConfig], Field(min_length=1)],
        sample_builder: HydraConfig[SampleTableBuilder],
    ) -> None:
        logger.debug("initializing dataset")

        super().__init__()

        _sample_builder = sample_builder.instantiate()
        samples: Mapping[str, pl.LazyFrame] = {}
        for input_id, input_cfg in inputs.items():
            with bound_contextvars(input_id=input_id):
                table = self._build_table(input_cfg)
                samples[input_id] = _sample_builder.build(table)
                logger.debug(
                    "built samples",
                    rows=table.select(pl.len()).collect().item(),
                    samples=samples[input_id].select(pl.len()).collect().item(),
                )

        input_id_enum = pl.Enum(sorted(samples))

        self._samples = (
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
            .collect()
            .rechunk()
        )

        self._frame_sources = (
            pl.LazyFrame(
                [
                    {
                        Column.input_id: input_id,
                        (k := "source"): [
                            source_cfg.model_dump(exclude={"reader"})
                            | {
                                "id": source_id,
                                "reader": source_cfg.reader.model_dump_json(
                                    by_alias=True
                                ),
                            }
                            for source_id, source_cfg in input_cfg.frame.items()
                        ],
                    }
                    for input_id, input_cfg in inputs.items()
                ],
                schema_overrides={Column.input_id: input_id_enum},
            )
            .explode(k)
            .unnest(k)
            .select(Column.input_id, pl.exclude(Column.input_id).name.prefix(f"{k}."))
            .collect()
            .rechunk()
        )

    @classmethod
    def _build_table(cls, sources: SourcesConfig) -> pl.LazyFrame:
        logger.debug("building table")

        match sources:
            case SourcesConfig(frame=frame_sources, table=None) if (
                len(frame_sources) == 1
            ):
                frame_source = mit.one(frame_sources.values())
                frame_reader = frame_source.reader.instantiate()
                frame_idxs = pl.Series(
                    name=frame_source.index_column,
                    values=frame_reader.get_available_indexes(),
                    dtype=pl.UInt32,
                ).sort()

                return pl.LazyFrame(frame_idxs)

            case SourcesConfig(
                frame=frame_sources, table=TableSourceConfig(builder=builder)
            ):
                table_builder = builder.instantiate()
                table = table_builder.build().lazy()
                schema = table.collect_schema()

                for frame_source_id, frame_source in frame_sources.items():
                    logger.debug("pruning table", frame_source=frame_source_id)
                    frame_reader = frame_source.reader.instantiate()
                    frame_idxs = pl.Series(
                        name=(col := frame_source.index_column),
                        values=frame_reader.get_available_indexes(),
                        dtype=schema[col],
                    ).sort()

                    table = table.join(
                        pl.LazyFrame(frame_idxs), on=frame_idxs.name, how="semi"
                    )

                return table

            case _:
                logger.error("not implemented")

                raise NotImplementedError

    @property
    def samples(self) -> pl.DataFrame:
        return self._samples

    @property
    def frame_sources(self) -> pl.DataFrame:
        return self._frame_sources

    @cache  # noqa: B019
    def _get_frame_reader(self, reader_json: str) -> FrameReader:  # noqa: PLR6301
        return HydraConfig[FrameReader].model_validate_json(reader_json).instantiate()

    def __getitems__(self, indexes: Sequence[int]) -> Batch:  # noqa: PLW3201
        samples = self.samples[indexes]
        batch_size = [samples.height]

        meta = BatchMeta(
            sample_idx=samples[Column.sample_idx].to_torch(),  # pyright: ignore[reportCallIssue]
            input_id=samples[Column.input_id].to_list(),  # pyright: ignore[reportCallIssue]
            batch_size=batch_size,  # pyright: ignore[reportCallIssue]
        )

        frame_source_idx_cols = self._frame_sources[Column.source_index_column].unique()

        frame_sources = (
            samples.lazy()
            .join(self._frame_sources.lazy(), on=Column.input_id, how="left")
            .with_columns(
                pl.coalesce(
                    pl.when(pl.col(Column.source_index_column) == idx_col).then(idx_col)
                    for idx_col in frame_source_idx_cols
                ).alias(Column.frame_idx)
            )
            .group_by(Column.source_id)
            .agg(Column.source_reader, Column.frame_idx)
        )

        frames = TensorDict(
            {
                row[Column.source_id]: torch.stack([
                    self._get_frame_reader(reader).read(frame_idxs)
                    for (reader, frame_idxs) in zip(
                        row[Column.source_reader], row[Column.frame_idx], strict=True
                    )
                ])
                for row in frame_sources.collect().iter_rows(named=True)
            },
            batch_size=batch_size,
        )

        table = TensorDict(
            samples.select(  # pyright: ignore[reportArgumentType]
                pl.exclude(Column.sample_idx, Column.input_id).to_physical()
            ).to_dict(as_series=False),
            batch_size=batch_size,
        )

        return Batch(meta=meta, frame=frames, table=table, batch_size=batch_size)  # pyright: ignore[reportCallIssue]

    def __len__(self) -> int:
        return len(self.samples)
