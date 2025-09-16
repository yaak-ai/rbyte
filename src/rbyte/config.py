from concurrent.futures import Executor
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Self, override

from hydra.utils import instantiate
from pipefunc import Pipeline
from pipefunc._pipeline._types import OUTPUT_TYPE
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    InstanceOf,
    TypeAdapter,
    model_validator,
)

from rbyte.types import TensorSource


class HydraConfig[T](BaseModel):
    target: ImportString[type[T]] = Field(
        serialization_alias="_target_", validation_alias="_target_"
    )
    recursive: bool = Field(alias="_recursive_", default=True)
    convert: Literal["none", "partial", "object", "all"] = Field(
        alias="_convert_", default="all"
    )
    partial: bool = Field(alias="_partial_", default=False)

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def instantiate(self, **kwargs: object) -> T:
        return instantiate(self.model_dump(by_alias=True), **kwargs)


class PickleableImportString[T](BaseModel):
    obj: ImportString[T]
    _path: str

    model_config = ConfigDict(extra="allow", frozen=True, validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, path: str) -> dict[str, str]:
        return {"obj": path, "_path": path}

    @override
    def __getstate__(self) -> dict[Any, Any]:
        state = deepcopy(super().__getstate__())
        state["__dict__"].pop("obj")

        return state

    @override
    def __setstate__(self, state: dict[Any, Any]) -> None:
        state["__dict__"]["obj"] = TypeAdapter(ImportString[T]).validate_python(
            state["__pydantic_extra__"]["_path"]
        )
        super().__setstate__(state)


class StreamConfig(BaseModel):
    index: str | tuple[str, ...]
    sources: dict[str, HydraConfig[TensorSource]]

    model_config = ConfigDict(extra="forbid")


type StreamsConfig = dict[str, StreamConfig]


class BasePipelineConfig(BaseModel):
    inputs: dict[str, list[Any]]
    run_folder: str | Path | None = None
    return_results: bool = True

    model_config = ConfigDict(extra="allow")

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
