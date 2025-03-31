from copy import deepcopy
from functools import cached_property
from typing import Any, ClassVar, Literal, override

from hydra.utils import instantiate
from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, Field, ImportString, TypeAdapter, model_validator


class BaseModel(_BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
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


class PickleableImportString[T](BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="allow", frozen=True, validate_assignment=True
    )

    obj: ImportString[T]
    _path: str

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
