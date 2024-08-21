from collections.abc import Sequence
from enum import StrEnum, unique

import polars as pl
import torch
from hydra.utils import instantiate
from polars._utils.getitem import (
    _select_rows_by_index,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
)
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tensordict import TensorDict
from torch.utils.data import Dataset as TorchDataset

from rbyte.batch import Batch, BatchMeta

from .config import DatasetConfig, FrameSourceConfig, SourcesConfig, TableSourceConfig

__all__ = ["Dataset"]

logger = get_logger(__name__)


@unique
class Column(StrEnum):
    input_id = "__input_id"
    sample_idx = "__sample_idx"
    frame_idx = "__frame_idx"
    source_id = "source.id"
    source_reader = "source.reader"
    source_index_column = "source.index_column"


class Dataset(TorchDataset[TensorDict]):
    def __init__(self, config: object) -> None:
        super().__init__()

        config = DatasetConfig.model_validate(config)

        samples: dict[str, pl.LazyFrame] = {}
        sample_builder = config.samples.builder.instantiate()
        for _input in config.inputs:
            with bound_contextvars(input=_input.id):
                table = self._build_table(_input.sources)
                samples[_input.id] = sample_builder.build(table)

                logger.debug(
                    "processed",
                    rows=table.select(pl.len()).collect().item(),
                    samples=samples[_input.id].select(pl.len()).collect().item(),
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
                        Column.input_id: _input.id,
                        (k := "source"): [
                            source.model_dump(by_alias=True)
                            for source in _input.sources.frame
                        ],
                    }
                    for _input in config.inputs
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
        match sources:
            case SourcesConfig(
                frame=[FrameSourceConfig(reader=reader, index_column=index_column)],
                table=None,
            ):
                frame_reader = reader.instantiate()
                frame_idxs = pl.Series(
                    name=index_column,
                    values=frame_reader.get_available_indices(),
                    dtype=pl.UInt32,
                )

                return pl.LazyFrame(frame_idxs)

            case SourcesConfig(
                frame=frame_sources, table=TableSourceConfig(path=path, builder=builder)
            ):
                table_builder = builder.instantiate()
                table_df = table_builder.build(path).lazy()
                schema = table_df.collect_schema()

                for frame_source in frame_sources:
                    frame_reader = frame_source.reader.instantiate()
                    frame_idxs = pl.Series(
                        name=(col := frame_source.index_column),
                        values=frame_reader.get_available_indices(),
                        dtype=schema[col],
                    )
                    table_df = table_df.join(
                        pl.LazyFrame(frame_idxs), on=frame_idxs.name, how="semi"
                    )

                return table_df

            case _:
                raise NotImplementedError

    @property
    def samples(self) -> pl.DataFrame:
        return self._samples

    @property
    def frame_sources(self) -> pl.DataFrame:
        return self._frame_sources

    def __getitems__(self, idxs: Sequence[int]) -> Batch:  # pyright: ignore[reportGeneralTypeIssues, reportUnknownParameterType]  # noqa: PLW3201
        samples = _select_rows_by_index(
            self.samples, pl.Series(values=idxs, dtype=pl.UInt32)
        )

        meta = BatchMeta(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            sample_idx=samples[Column.sample_idx].to_torch(),  # pyright: ignore[reportCallIssue]
            input_id=samples[Column.input_id].to_list(),  # pyright: ignore[reportCallIssue]
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
                    instantiate(reader).read(frame_idxs)
                    for (reader, frame_idxs) in zip(
                        row[Column.source_reader], row[Column.frame_idx], strict=True
                    )
                ])
                for row in frame_sources.collect().iter_rows(named=True)
            },
            batch_size=[len(idxs)],
        )

        table = TensorDict(
            samples.select(  # pyright: ignore[reportArgumentType]
                pl.exclude(Column.sample_idx, Column.input_id)
                .arr.to_list()
                .to_physical()
            ).to_dict()
        )

        return Batch(meta=meta, frame=frames, table=table).auto_batch_size_(1)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportUnknownMemberType]

    def __len__(self) -> int:
        return len(self.samples)
