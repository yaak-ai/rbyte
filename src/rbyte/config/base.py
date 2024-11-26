from functools import cached_property
from typing import ClassVar, Literal

from hydra.utils import instantiate
from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, Field, ImportString, model_validator
from pydantic import RootModel as _RootModel


class BaseModel(_BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        ignored_types=(cached_property,),
    )


class RootModel[T](_RootModel[T]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )


class HydraConfig[T](BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    target: ImportString[type[T]] = Field(alias="_target_")
    recursive: bool = Field(alias="_recursive_", default=True)
    convert: Literal["none", "partial", "object", "all"] = Field(
        alias="_convert_", default="all"
    )
    partial: bool = Field(alias="_partial_", default=False)

    def instantiate(self, **kwargs: object) -> T:
        return instantiate(self.model_dump(by_alias=True), **kwargs)

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: object) -> object:
        match data:
            case str():
                return {"_target_": data}

            case _:
                return data
