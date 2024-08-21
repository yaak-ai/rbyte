from functools import cached_property
from typing import TypeVar

from hydra.utils import instantiate
from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, Field, ImportString, field_serializer


class BaseModel(_BaseModel):
    class Config:
        frozen = True
        extra = "forbid"
        validate_assignment = True


T = TypeVar("T")


class HydraConfig[T](BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="allow", ignored_types=(cached_property,)
    )

    target: ImportString[type[T]] = Field(alias="_target_")

    def instantiate(self, **kwargs: object) -> T:
        return instantiate(self.model_dump(by_alias=True), **kwargs)

    @field_serializer("target")
    @staticmethod
    def serialize_target(v: object) -> str:
        return ImportString._serialize(v)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]  # noqa: SLF001
