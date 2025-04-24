from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import cache, cached_property
from math import prod
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Protocol,
    cast,
    override,
    runtime_checkable,
)

import more_itertools as mit
import rerun as rr
import torch
from hydra.utils import get_method, instantiate
from pydantic import BeforeValidator, ConfigDict, Field, RootModel, validate_call
from rerun._send_columns import TimeColumnLike as _TimeColumnLike  # noqa: PLC2701
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tensordict import TensorClass, TensorDict

from rbyte.config.base import BaseModel, HydraConfig

from .base import Logger

logger = get_logger(__name__)


@runtime_checkable
class TimeColumnLike(_TimeColumnLike, Protocol): ...


class IndexSchemaItem(HydraConfig[TimeColumnLike]):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    times: str | tuple[str, ...]


class MethodHydraConfig[T](BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    target: Annotated[Callable[..., T], BeforeValidator(get_method)] = Field(
        alias="_target_"
    )

    def instantiate(self, **kwargs: object) -> T:
        return instantiate(self.model_dump(by_alias=True), **kwargs)


class AsComponentsConfig(MethodHydraConfig[rr.AsComponents]): ...


class ComponentColumnListConfig(MethodHydraConfig[rr.ComponentColumnList]):
    __pydantic_extra__: dict[str, str | tuple[str, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]


class StaticSchemaItem(BaseModel):
    static: Literal[True]
    entity: AsComponentsConfig


class Schema(
    RootModel[
        dict[
            str,
            IndexSchemaItem
            | StaticSchemaItem
            | ComponentColumnListConfig
            | Sequence[StaticSchemaItem | ComponentColumnListConfig],
        ]
    ]
):
    @cached_property
    def indexes(self) -> dict[str, IndexSchemaItem]:
        return {k: v for k, v in self.root.items() if isinstance(v, IndexSchemaItem)}

    @cached_property
    def static(self) -> dict[str, Sequence[StaticSchemaItem]]:
        result: dict[str, list[StaticSchemaItem]] = defaultdict(list)

        for k, v in self.root.items():
            match v:
                case StaticSchemaItem():
                    result[k].append(v)

                case Sequence():
                    result[k].extend(x for x in v if isinstance(x, StaticSchemaItem))

                case _:
                    pass

        return {k: v for k, v in result.items() if v}

    @cached_property
    def columns(self) -> dict[str, Sequence[ComponentColumnListConfig]]:
        result: dict[str, list[ComponentColumnListConfig]] = defaultdict(list)

        for k, v in self.root.items():
            match v:
                case ComponentColumnListConfig():
                    result[k].append(v)

                case Sequence():
                    result[k].extend(
                        x for x in v if isinstance(x, ComponentColumnListConfig)
                    )

                case _:
                    pass

        return {k: v for k, v in result.items() if v}


class RerunLogger(Logger[TensorDict | TensorClass]):
    @validate_call
    def __init__(
        self,
        *,
        application_id: str | tuple[str, ...],
        schema: Schema,
        spawn: bool = True,
        port: int = 9876,
    ) -> None:
        super().__init__()

        self._application_id: str | tuple[str, ...] = application_id
        self._schema: Schema = schema
        self._spawn: bool = spawn
        self._port: int = port

    @cache  # noqa: B019
    def _get_recording(self, application_id: str) -> rr.RecordingStream:
        recording = rr.RecordingStream(application_id, make_default=True)
        if self._spawn:
            recording.spawn(port=self._port)

        for path, items in self._schema.static.items():
            rr.log(
                path,
                *(item.entity.instantiate() for item in items),
                static=True,
                recording=recording,
            )

        return recording

    @classmethod
    def _build_columns(  # noqa: C901, PLR0912
        cls, config: ComponentColumnListConfig, data: TensorDict
    ) -> rr.ComponentColumnList:
        kwargs = TensorDict({k: data[v] for k, v in config.__pydantic_extra__.items()})

        with bound_contextvars(target=config.target):
            match cast(Any, config.target):
                case rr.Image.columns | rr.DepthImage.columns:
                    match (tensor := kwargs[(key := "buffer")]).shape:
                        case (*batch_dims, _h, _w, 3):
                            pass

                        case (*batch_dims, 3, _h, _w):
                            tensor = tensor.permute(*range(len(batch_dims)), -2, -1, -3)

                        case (*batch_dims, _d, _w):
                            pass

                        case shape:
                            logger.error(
                                (msg := "shape not supported"), key=key, shape=shape
                            )
                            raise NotImplementedError(msg)

                    kwargs[key] = tensor.reshape(prod(batch_dims), -1).view(torch.uint8)

                    return config.instantiate(**kwargs.cpu().numpy())  # pyright: ignore[reportUnknownMemberType]

                case rr.Points2D.columns:
                    match (tensor := kwargs[key := "positions"]).shape:
                        case (2,):
                            return config.instantiate(**kwargs.cpu().numpy())  # pyright: ignore[reportUnknownMemberType]

                        case (*batch_dims, n, 2):
                            kwargs[key] = tensor.view(-1, 2)
                            return config.instantiate(**kwargs.cpu().numpy()).partition(  # pyright: ignore[reportUnknownMemberType]
                                [n] * prod(batch_dims)
                            )

                        case shape:
                            logger.error(
                                (msg := "shape not supported"), key=key, shape=shape
                            )
                            raise NotImplementedError(msg)

                case rr.Points3D.columns:
                    match (tensor := kwargs[key := "positions"]).shape:
                        case (3,):
                            kwargs[key] = tensor.view(-1, 3)
                            return config.instantiate(**kwargs.cpu().numpy())  # pyright: ignore[reportUnknownMemberType]

                        case (*batch_dims, n, 3):
                            kwargs[key] = tensor.view(-1, 3)
                            return config.instantiate(**kwargs.cpu().numpy()).partition(  # pyright: ignore[reportUnknownMemberType]
                                [n] * prod(batch_dims)
                            )

                        case shape:
                            logger.error(
                                (msg := "shape not supported"), key=key, shape=shape
                            )
                            raise NotImplementedError(msg)

                case _:
                    return config.instantiate(**kwargs.cpu().numpy())  # pyright: ignore[reportUnknownMemberType]

    @override
    def log(self, data: TensorDict | TensorClass) -> None:
        data = data.to_tensordict()

        match application_id := self._application_id:
            case str():
                with self._get_recording(application_id):
                    self._log(data)

            case tuple():
                for application_id_elem, data_elem in zip(  # pyright: ignore[reportUnknownVariableType]
                    map(str, data[application_id]), data, strict=True
                ):
                    with self._get_recording(application_id_elem):
                        self._log(data_elem)  # pyright: ignore[reportUnknownArgumentType]

    def _log(self, data: TensorDict) -> None:
        indexes = [
            config.instantiate(
                timeline=timeline,
                times=torch.atleast_1d(data[config.times].flatten()).cpu().numpy(),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            )
            for timeline, config in self._schema.indexes.items()
        ]

        for entity_path, configs in self._schema.columns.items():
            with bound_contextvars(entity_path=entity_path):
                columns = [self._build_columns(config, data) for config in configs]

                try:
                    rr.send_columns(
                        entity_path=entity_path,
                        indexes=indexes,
                        columns=mit.flatten(columns),
                        strict=True,
                    )
                except Exception:
                    logger.exception("rr.send_columns failed")

                    raise
