from collections.abc import Mapping, Sequence
from enum import StrEnum, unique
from functools import cache
from typing import Annotated, Literal, override

import polars as pl
import torch
from hydra.utils import instantiate
from pipefunc import Pipeline
from pydantic import ConfigDict, Field, StringConstraints, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
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
    source: HydraConfig[TensorSource]
    index_column: str


class PipelineConfig(BaseModel):
    pipeline: HydraConfig[Pipeline]
    output_name: str | None = None
    kwargs: dict[str, object] = Field(default_factory=dict)


class InputConfig(BaseModel):
    sources: Mapping[Id, SourceConfig] = Field(min_length=1)
    samples: PipelineConfig


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
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self, inputs: Annotated[Mapping[Id, InputConfig], Field(min_length=1)]
    ) -> None:
        logger.debug("initializing dataset")

        super().__init__()

        samples: Mapping[str, pl.DataFrame] = {}
        for input_id, input_cfg in inputs.items():
            with bound_contextvars(input_id=input_id):
                samples_cfg = input_cfg.samples
                pipeline = samples_cfg.pipeline.instantiate()
                output_name = (
                    samples_cfg.output_name or pipeline.unique_leaf_node.output_name  # pyright: ignore[reportUnknownMemberType]
                )
                kwargs = instantiate(
                    samples_cfg.kwargs, _recursive_=True, _convert_="all"
                )
                samples[input_id] = pipeline.run(output_name=output_name, kwargs=kwargs)
                logger.debug(
                    "built samples",
                    columns=samples[input_id].columns,
                    len=len(samples[input_id]),
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

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, validate_default=False)
    )
    def get_batch(
        self,
        index: int | Sequence[int] | slice | range,
        *,
        keys: BatchKeys = BATCH_KEYS_DEFAULT,
    ) -> Batch:
        subkeys: Mapping[Literal["data", "meta"], set[_ALL_TYPE | str]] = {
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
            source_idx_cols = self._sources[Column.source_index_column].unique()
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
                        self._get_source(source)[idxs]
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
