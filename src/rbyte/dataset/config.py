from collections.abc import Mapping
from functools import cached_property
from operator import attrgetter
from typing import Annotated

from more_itertools import all_unique
from pydantic import Field, FilePath, StringConstraints, computed_field, field_validator

from rbyte.config.base import BaseModel, HydraConfig
from rbyte.io.frame.base import FrameReader
from rbyte.io.table.base import TableBuilder
from rbyte.sample.base import SampleTableBuilder


class FrameSourceConfig(BaseModel):
    id: str
    reader: HydraConfig[FrameReader]
    index_column: str


class TableSourceConfig(BaseModel):
    path: FilePath
    builder: HydraConfig[TableBuilder]


class SourcesConfig(BaseModel):
    frame: tuple[FrameSourceConfig, ...] = Field(min_length=1)
    table: TableSourceConfig | None = None

    @field_validator("frame", mode="after")
    @classmethod
    def validate_frame_sources(
        cls, sources: tuple[FrameSourceConfig, ...]
    ) -> tuple[FrameSourceConfig, ...]:
        if not all_unique(sources, key=attrgetter("id")):
            msg = "frame source ids not unique"
            raise ValueError(msg)

        return sources


class InputConfig(BaseModel):
    id: Annotated[
        str, StringConstraints(strip_whitespace=True, pattern=r"^[\x00-\x7F]+$")
    ]
    sources: SourcesConfig


class SamplesConfig(BaseModel):
    builder: HydraConfig[SampleTableBuilder]


class DatasetConfig(BaseModel):
    inputs: tuple[InputConfig, ...] = Field(min_length=1)
    samples: SamplesConfig

    @field_validator("inputs", mode="after")
    @classmethod
    def validate_inputs(
        cls, inputs: tuple[InputConfig, ...]
    ) -> tuple[InputConfig, ...]:
        if not all_unique(inputs, key=attrgetter("id")):
            msg = "input ids not unique"
            raise ValueError(msg)

        return inputs

    @computed_field
    @cached_property
    def inputs_by_id(self) -> Mapping[str, InputConfig]:
        return {x.id: x for x in self.inputs}
