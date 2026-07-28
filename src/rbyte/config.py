from collections.abc import Callable
from concurrent.futures import Executor
from inspect import ismethod
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from hydra.utils import get_object, instantiate
from pipefunc import Pipeline
from pipefunc._pipeline._types import OUTPUT_TYPE
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    InstanceOf,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from rbyte.streams.base import StreamSource


class HydraConfig[T](BaseModel):
    target: type[T] | Callable[..., T] = Field(
        serialization_alias="_target_", validation_alias="_target_"
    )
    recursive: bool = Field(alias="_recursive_", default=True)
    convert: Literal["none", "partial", "object", "all"] = Field(
        alias="_convert_", default="all"
    )
    partial: bool = Field(alias="_partial_", default=False)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="allow",
        validate_assignment=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    def instantiate(self, **kwargs: object) -> T:
        return instantiate(self.model_dump(by_alias=True), **kwargs)

    @field_validator("target", mode="before")
    @classmethod
    def resolve_target(cls, target: object) -> object:
        if isinstance(target, str):
            try:
                target = get_object(target)
            except Exception as e:
                msg = f"unable to resolve Hydra target {target!r}"
                raise ValueError(msg) from e

        return target

    @field_serializer("target", when_used="always")
    def serialize_target(self, target: object, _info: SerializationInfo) -> str:  # ruff:ignore[no-self-use]
        if (
            ismethod(target)
            and isinstance(target.__self__, type)
            and isinstance(target.__name__, str)
        ):
            return f"{target.__self__.__module__}.{target.__self__.__qualname__}.{target.__name__}"  # ruff:ignore[line-too-long]

        if isinstance(target, type) or callable(target):
            module = getattr(target, "__module__", None)
            qualname = getattr(target, "__qualname__", None)
            if isinstance(module, str) and isinstance(qualname, str):
                return f"{module}.{qualname}"

        return ImportString._serialize(target)  # ruff:ignore[private-member-access]


class StreamConfig(BaseModel):
    index: str | tuple[str, ...]
    sources: dict[str, HydraConfig[StreamSource]]

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
