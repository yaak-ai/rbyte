from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from math import prod
from typing import Annotated, Any, Generic, Literal, TypeVar, Union, cast

import rerun as rr
import rerun.blueprint as rrb
import torch
from cachetools import Cache, cachedmethod
from einops import rearrange
from hydra.utils import get_method
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    InstanceOf,
    RootModel,
    validate_call,
)
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tensordict import TensorClass, TensorDict
from torch import Tensor, uint8
from typing_extensions import override

from rbyte.config import HydraConfig

from .base import Logger

logger = get_logger(__name__)


T = TypeVar("T")


class MethodHydraConfig(HydraConfig[T], Generic[T]):
    target: Annotated[Callable[..., T], BeforeValidator(get_method)] = Field(
        alias="_target_"
    )


class TimeColumnSchemaItem(HydraConfig[rr.TimeColumn]):
    dtype: str | None = Field(default=None, exclude=True)


class StaticSchemaItem(MethodHydraConfig[rr.AsComponents]):
    static: Literal[True] = Field(exclude=True)


Indices = Union[tuple[int, ...], tuple[str, ...]]


class ComponentColumnSchemaItem(MethodHydraConfig[rr.ComponentColumnList]):
    indices: Indices | None = Field(default=None, exclude=True)


class Schema(
    RootModel[
        dict[
            str,
            TimeColumnSchemaItem
            | Sequence[StaticSchemaItem | ComponentColumnSchemaItem],
        ]
    ]
):
    @cached_property
    def time_columns(self) -> dict[str, TimeColumnSchemaItem]:
        return {
            k: v for k, v in self.root.items() if isinstance(v, TimeColumnSchemaItem)
        }

    @cached_property
    def static(self) -> dict[str, Sequence[StaticSchemaItem]]:
        return {
            k: items
            for k, v in self.root.items()
            if isinstance(v, Sequence)
            and (items := [item for item in v if isinstance(item, StaticSchemaItem)])
        }

    @cached_property
    def component_columns(self) -> dict[str, Sequence[ComponentColumnSchemaItem]]:
        return {
            k: items
            for k, v in self.root.items()
            if isinstance(v, Sequence)
            and (
                items := [
                    item for item in v if isinstance(item, ComponentColumnSchemaItem)
                ]
            )
        }


class RerunLogger(Logger[TensorDict | TensorClass]):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        application_id: str,
        recording_name: str | tuple[str, ...],
        schema: Schema,
        spawn: bool = True,
        port: int = 9876,
        blueprint: InstanceOf[rrb.BlueprintLike]
        | Annotated[
            MethodHydraConfig[rrb.BlueprintLike],
            AfterValidator(MethodHydraConfig.instantiate),
        ]
        | None = None,
    ) -> None:
        super().__init__()

        self._application_id: str = application_id
        self._recording_name: str | tuple[str, ...] = recording_name
        self._schema: Schema = schema
        self._spawn: bool = spawn
        self._port: int = port
        self._blueprint: rrb.BlueprintLike | None = blueprint

        self._recordings: Cache[str, rr.RecordingStream] = Cache(maxsize=math.inf)

    @property
    def recordings(self) -> Cache[str, rr.RecordingStream]:
        return self._recordings

    @cachedmethod(lambda self: self._recordings)
    def _get_recording(self, name: str) -> rr.RecordingStream:
        recording = rr.RecordingStream(self._application_id)
        if self._spawn:
            recording.spawn(port=self._port, default_blueprint=self._blueprint)

        recording.send_recording_name(name)

        for path, items in self._schema.static.items():
            recording.log(path, *(item.instantiate() for item in items), static=True)

        return recording

    def _build_time_columns(
        self, data: TensorDict, indices: Indices | None = None
    ) -> Iterable[rr.TimeColumn]:
        column_indices = (
            data[*indices]
            if isinstance(indices, tuple)
            and all(isinstance(idx, str) for idx in indices)
            else indices
        )

        for timeline, config in self._schema.time_columns.items():
            kwargs: dict[str, Any] = {}
            for k, k_data in config.model_extra.items():
                v = (
                    torch
                    .atleast_1d(data[*k_data][column_indices].flatten())  # ty: ignore[invalid-argument-type]
                    .cpu()
                    .numpy()
                )
                kwargs[k] = v if (dtype := config.dtype) is None else v.astype(dtype)

            yield config.instantiate(timeline=timeline, **kwargs)

    @classmethod
    def _build_component_columns(  # noqa: C901, PLR0912
        cls, config: ComponentColumnSchemaItem, data: TensorDict
    ) -> rr.ComponentColumnList:
        kwargs = TensorDict({k: data[*v] for k, v in config.model_extra.items()})  # ty: ignore[possibly-missing-attribute]
        lengths: list[int] | None = None

        with bound_contextvars(target=config.target):
            match cast(Any, config.target):
                case rr.Image.columns:
                    match tensor := kwargs.get(key := "buffer"):
                        case Tensor(shape=(*_batch_dims, 3, _h, _w)):
                            kwargs[key] = rearrange(
                                tensor, "... c h w -> (...) (h w c)"
                            ).view(uint8)

                        case Tensor(shape=(*_batch_dims, _h, _w, 3)):
                            kwargs[key] = rearrange(
                                tensor, "... h w c -> (...) (h w c)"
                            ).view(uint8)

                        case _:
                            logger.error(
                                (msg := "shape not supported"),
                                key=key,
                                shape=tensor.shape,
                            )
                            raise NotImplementedError(msg)

                case rr.DepthImage.columns:
                    match tensor := kwargs.get(key := "buffer"):
                        case Tensor(shape=(*_, _h, _w)):
                            kwargs[key] = rearrange(
                                tensor, "... h w -> (...) (h w)"
                            ).view(torch.uint8)

                        case _:
                            logger.error(
                                (msg := "shape not supported"),
                                key=key,
                                shape=tensor.shape,
                            )
                            raise NotImplementedError(msg)

                case rr.Points2D.columns:
                    match tensor := kwargs.get(key := "positions"):
                        case Tensor(shape=(2,)):
                            pass

                        case Tensor(shape=(*batch_dims, n, 2)):
                            kwargs[key] = rearrange(tensor, "... n d -> (... n) d")
                            lengths = [n] * prod(batch_dims)

                        case _:
                            logger.error(
                                (msg := "shape not supported"),
                                key=key,
                                shape=tensor.shape,
                            )
                            raise NotImplementedError(msg)

                case rr.Points3D.columns:
                    match tensor := kwargs.get(key := "positions"):
                        case Tensor(shape=(3,)):
                            pass

                        case Tensor(shape=(*batch_dims, n, 3)):
                            kwargs[key] = rearrange(tensor, "... n d -> (... n) d")
                            lengths = [n] * prod(batch_dims)

                        case _:
                            logger.error(
                                (msg := "shape not supported"),
                                key=key,
                                shape=tensor.shape,
                            )
                            raise NotImplementedError(msg)

                case _:
                    pass

        return config.instantiate(**kwargs.cpu().numpy()).partition(lengths)  # ty: ignore[invalid-argument-type]

    @override
    def log(self, data: TensorDict | TensorClass) -> None:
        data = data.to_tensordict()

        match recording_name := self._recording_name:
            case str():
                with self._get_recording(recording_name):
                    self._log(data)  # ty: ignore[invalid-argument-type]

            case tuple():
                for data_elem in data:
                    with self._get_recording(data_elem[recording_name]):
                        self._log(data_elem)

    def _log(self, data: TensorDict) -> None:
        time_columns: dict[Indices | None, list[rr.TimeColumn]] = {}

        for (
            entity_path,
            component_column_configs,
        ) in self._schema.component_columns.items():
            for column_config in component_column_configs:
                component_columns = self._build_component_columns(column_config, data)

                if (indices := column_config.indices) not in time_columns:
                    time_columns[indices] = list(
                        self._build_time_columns(data, indices)
                    )

                rr.send_columns(
                    entity_path=entity_path,
                    indexes=time_columns[indices],
                    columns=component_columns,
                )
