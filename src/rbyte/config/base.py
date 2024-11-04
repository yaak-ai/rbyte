from functools import cached_property
from typing import ClassVar, Literal, TypeVar

from hydra.utils import instantiate
from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, Field, ImportString, field_serializer, model_validator


class BaseModel(_BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        ignored_types=(cached_property,),
    )


T = TypeVar("T")


class HydraConfig[T](BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    target: ImportString[type[T]] = Field(alias="_target_")
    recursive: bool = Field(alias="_recursive_", default=True)
    convert: Literal["none", "partial", "object", "all"] = Field(
        alias="_convert_", default="none"
    )
    partial: bool = Field(alias="_partial_", default=False)

    def instantiate(self, **kwargs: object) -> T:
        return instantiate(self.model_dump(by_alias=True), **kwargs)

    @field_serializer("target")
    @staticmethod
    def serialize_target(v: object) -> str:
        return ImportString._serialize(v)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]  # noqa: SLF001

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: object) -> object:
        match data:
            case str():
                return {"_target_": data}

            case _:
                return data
