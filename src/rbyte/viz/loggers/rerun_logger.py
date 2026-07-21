import ast
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from math import prod
from types import EllipsisType
from typing import Annotated, Any, Literal, override
from uuid import UUID

import more_itertools as mit
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.nn.functional as F  # ruff:ignore[lowercase-imported-as-non-lowercase]
from einops import rearrange
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    RootModel,
    model_validator,
    validate_call,
)
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tensordict import TensorClass, TensorDict
from torch import Tensor, uint8

from rbyte.config import HydraConfig

from .base import Logger

logger = get_logger(__name__)


type _TensorIndex = int | slice | EllipsisType | tuple[_TensorIndex, ...] | None


class TensorIndex(RootModel[object]):
    @property
    def raw(self) -> _TensorIndex:
        return self.root  # ty:ignore[invalid-return-type]

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, value: object) -> _TensorIndex:
        if isinstance(value, cls):
            return value.raw

        return cls.parse(value)

    @classmethod
    def parse(cls, value: object) -> _TensorIndex:
        match value:
            case None:
                return ()

            case str():
                return cls._parse_string(value)

            case _:
                return cls._validate_value(value)

    @classmethod
    def _parse_string(cls, index: str) -> _TensorIndex:
        try:
            expression = ast.parse(f"_value{index}", mode="eval")
        except SyntaxError as exc:
            msg = f"unsupported tensor selector index syntax: {index!r}"
            raise ValueError(msg) from exc

        match expression.body:
            case ast.Subscript(value=ast.Name(id="_value"), slice=slice_node):
                return cls._parse_ast(slice_node, index)

            case _:
                msg = f"tensor selector index must be a bracket expression: {index!r}"
                raise ValueError(msg)

    @classmethod
    def _validate_value(cls, value: object) -> _TensorIndex:
        match value:
            case None | int() | EllipsisType():
                return value

            case slice(start=(int() | None), stop=(int() | None), step=(int() | None)):
                return value

            case tuple():
                return tuple(cls._validate_value(x) for x in value)

            case _:
                msg = f"unsupported tensor selection index value: {value!r}"
                raise ValueError(msg)

    @classmethod
    def _parse_ast(
        cls, node: ast.expr | None, source: str, *, slice_bound: bool = False
    ) -> _TensorIndex:
        if node is None:
            if slice_bound:
                return None
            node = ast.Constant(value=None)
            msg = (
                f"unsupported tensor selection index syntax: {source!r}; "
                f"node={ast.dump(node, include_attributes=False)}"
            )
            raise ValueError(msg)

        if (value := cls._parse_ast_int(node)) is not None:
            return value

        if slice_bound:
            msg = (
                f"unsupported tensor selection index syntax: {source!r}; "
                f"node={ast.dump(node, include_attributes=False)}"
            )
            raise ValueError(msg)

        match node:
            case ast.Tuple(elts=elts):
                result = tuple(cls._parse_ast(elt, source) for elt in elts)

            case ast.Slice(lower=lower, upper=upper, step=step):
                result = slice(
                    cls._parse_ast(lower, source, slice_bound=True),
                    cls._parse_ast(upper, source, slice_bound=True),
                    cls._parse_ast(step, source, slice_bound=True),
                )

            case ast.Constant(value=None):
                result = None

            case ast.Constant(value=value) if value is Ellipsis:
                result = Ellipsis

            case ast.Name(id="Ellipsis"):
                result = Ellipsis

            case _:
                msg = (
                    f"unsupported tensor selection index syntax: {source!r}; "
                    f"node={ast.dump(node, include_attributes=False)}"
                )
                raise ValueError(msg)

        return result

    @staticmethod
    def _parse_ast_int(node: ast.expr) -> int | None:
        match node:
            case ast.Constant(value=value) if type(value) is int:
                return value

            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=value)) if (
                type(value) is int
            ):
                return -value

            case _:
                return None


class TensorSelector(BaseModel):
    path: tuple[str, ...]
    index: TensorIndex = Field(default_factory=lambda: TensorIndex.model_validate(None))

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def select(self, data: TensorDict) -> Tensor:
        return data[*self.path][self.index.raw]  # ty:ignore[invalid-argument-type, invalid-return-type]

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, value: object) -> object:
        if isinstance(value, list | tuple):
            return {"path": value}

        return value


class TimeColumnSchemaItem(HydraConfig[rr.TimeColumn]):
    columns: dict[str, TensorSelector] = Field(exclude=True)
    dtype: str | None = Field(default=None, exclude=True)

    model_config = ConfigDict(extra="forbid")


class StaticSchemaItem(HydraConfig[rr.AsComponents]):
    static: Literal[True] = Field(exclude=True)
    fields: dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(extra="forbid")


class DynamicTimeIndex(BaseModel):
    path: tuple[str, ...]

    model_config = ConfigDict(extra="forbid")


class StaticTimeIndex(BaseModel):
    index: TensorIndex

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


type TimeIndex = DynamicTimeIndex | StaticTimeIndex


class ComponentColumnSchemaItem(HydraConfig[rr.ComponentColumnList]):
    columns: dict[str, TensorSelector] = Field(exclude=True)
    time_index: TimeIndex | None = Field(default=None, exclude=True)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


@dataclass(frozen=True)
class TimeColumnBundle:
    columns: list[rr.TimeColumn]
    row_count: int


type NormalizedTensorIndex = int | tuple[object, ...]
type TimeIndexCacheKey = tuple[tuple[str, ...] | None, NormalizedTensorIndex]


def _normalize_tensor_index(index: _TensorIndex) -> NormalizedTensorIndex:
    match index:
        case int() as value:
            return value

        case slice(start=start, stop=stop, step=step):
            return ("slice", start, stop, step)

        case EllipsisType():
            return ("ellipsis",)

        case None:
            return ("none",)

        case tuple() as values:
            return tuple(_normalize_tensor_index(value) for value in values)

    msg = f"unsupported tensor index: {index!r}"
    raise TypeError(msg)


def _time_index_cache_key(time_index: TimeIndex | None) -> TimeIndexCacheKey:
    match time_index:
        case None:
            return (None, ())

        case DynamicTimeIndex(path=path):
            return (path, ())

        case StaticTimeIndex(index=index):
            return (None, _normalize_tensor_index(index.raw))


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
    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        application_id: str,
        recording_id: str | UUID | None = None,
        recording_name: str | tuple[str, ...],
        entity_path_format: str = "{}",
        schema: Schema,
        spawn: bool = True,
        port: int = 9876,
        blueprint: InstanceOf[rrb.BlueprintLike]
        | Annotated[
            HydraConfig[rrb.BlueprintLike], AfterValidator(HydraConfig.instantiate)
        ]
        | None = None,
    ) -> None:
        super().__init__()

        self._application_id = application_id
        self._recording_id = recording_id
        self._recording_name = recording_name
        self._entity_path_format = entity_path_format
        self._schema = schema
        self._spawn = spawn
        self._port = port
        self._blueprint: rrb.BlueprintLike | None = blueprint  # ty:ignore[invalid-assignment]

        self._recordings: dict[str, rr.RecordingStream] = {}

    @property
    def recordings(self) -> dict[str, rr.RecordingStream]:
        return self._recordings

    def _entity_path(self, path: str) -> str:
        return self._entity_path_format.format(path)

    def _get_recording(self, name: str) -> rr.RecordingStream:
        try:
            return self._recordings[name]
        except KeyError:
            pass

        recording = rr.RecordingStream(
            application_id=self._application_id, recording_id=self._recording_id
        )
        if self._spawn:
            recording.spawn(port=self._port, default_blueprint=self._blueprint)

        recording.send_recording_name(name)

        for path, items in self._schema.static.items():
            recording.log(
                self._entity_path(path),
                *(item.instantiate(**item.fields) for item in items),
                static=True,
            )

        self._recordings[name] = recording

        return recording

    def _build_time_columns(
        self, data: TensorDict, time_index: TimeIndex | None = None
    ) -> TimeColumnBundle:
        columns: list[rr.TimeColumn] = []
        row_counts: set[int] = set()
        time_columns = self._schema.time_columns

        for timeline, config in time_columns.items():
            kwargs: dict[str, Any] = {}
            for k, selector in config.columns.items():
                match time_index:
                    case DynamicTimeIndex(path=path):
                        values = selector.select(data)[
                            TensorSelector(path=path).select(data)
                        ]
                    case StaticTimeIndex(index=index):
                        values = selector.select(data)[index.raw]
                    case _:
                        values = selector.select(data)

                flattened_values = torch.atleast_1d(values.cpu().flatten())
                row_counts.add(flattened_values.numel())
                v = flattened_values.numpy()
                kwargs[k] = v if (dtype := config.dtype) is None else v.astype(dtype)

            columns.append(config.instantiate(timeline=timeline, **kwargs))

        row_count = mit.only(row_counts)
        if row_count is None:
            msg = "time columns must contain at least one tensor selection"
            raise ValueError(msg)

        return TimeColumnBundle(columns=columns, row_count=row_count)

    @classmethod
    def _build_component_columns(  # ruff:ignore[complex-structure, too-many-branches, too-many-statements]
        cls,
        config: ComponentColumnSchemaItem,
        data: TensorDict,
        row_count: int | None = None,
    ) -> rr.ComponentColumnList:
        kwargs = TensorDict({
            k: selector.select(data) for k, selector in config.columns.items()
        })
        lengths: list[int] | None = None

        with bound_contextvars(target=config.target):
            match config.target:
                case rr.Image.columns:
                    match value := kwargs[key := "buffer"]:
                        case Tensor(shape=(*_batch_dims, 3, _h, _w)):
                            kwargs[key] = rearrange(
                                value, "... c h w -> (...) (h w c)"
                            ).view(uint8)

                        case Tensor(shape=(*_batch_dims, _h, _w, 3)):
                            kwargs[key] = rearrange(
                                value, "... h w c -> (...) (h w c)"
                            ).view(uint8)

                        case _:
                            logger.error(
                                (msg := "shape not supported"),
                                key=key,
                                shape=value.shape,
                            )
                            raise NotImplementedError(msg)

                case rr.DepthImage.columns:
                    match value := kwargs[key := "buffer"]:
                        case Tensor(shape=(*_, _h, _w)):
                            kwargs[key] = rearrange(
                                value, "... h w -> (...) (h w)"
                            ).view(torch.uint8)

                        case _:
                            logger.error(
                                (msg := "shape not supported"),
                                key=key,
                                shape=value.shape,
                            )
                            raise NotImplementedError(msg)

                case rr.Points2D.columns:
                    match value := kwargs[key := "positions"]:
                        case Tensor(shape=(2,)):
                            pass

                        case Tensor(shape=(*batch_dims, points, 2)):
                            kwargs[key] = rearrange(value, "... n d -> (... n) d")
                            lengths = partition_lengths(
                                batch_dims,
                                instances_per_row=points,
                                row_count=row_count,
                            )

                        case _:
                            logger.error(
                                (msg := "shape not supported"),
                                key=key,
                                shape=value.shape,
                            )
                            raise NotImplementedError(msg)

                case rr.LineStrips3D.columns:
                    match tensor := atleast_nd_left(
                        value := kwargs[key := "strips"],  # ty:ignore[invalid-argument-type]
                        3,
                    ):
                        case Tensor(shape=(*batch_dims, points, _, 2 | 3 as dim)):
                            kwargs[key] = rearrange(
                                # https://github.com/rerun-io/rerun/issues/1387
                                F.pad(tensor, (0, 3 - dim), value=0),
                                "... segments dim -> (...) segments dim",
                            )
                            lengths = partition_lengths(
                                batch_dims,
                                instances_per_row=points,
                                row_count=row_count,
                            )

                        case _:
                            logger.error("not implemented", key=key, value=value)
                            raise NotImplementedError

                case rr.Points3D.columns:
                    match tensor := atleast_nd_left(
                        value := kwargs[key := "positions"],  # ty:ignore[invalid-argument-type]
                        3,
                    ):
                        case Tensor(shape=(*batch_dims, points, 2 | 3 as dim)):
                            kwargs[key] = rearrange(
                                # https://github.com/rerun-io/rerun/issues/1387
                                F.pad(tensor, (0, 3 - dim), value=0),
                                "... points dim -> (...) points dim",
                            )
                            lengths = partition_lengths(
                                batch_dims,
                                instances_per_row=points,
                                row_count=row_count,
                            )

                        case _:
                            logger.error("not implemented", key=key, value=value)
                            raise NotImplementedError

                case rr.GeoPoints.columns:
                    match value := kwargs[key := "positions"]:
                        case Tensor(shape=(2,)):
                            pass

                        case Tensor(shape=(*batch_dims, points, 2)):
                            kwargs[key] = rearrange(value, "... n d -> (... n) d")
                            lengths = partition_lengths(
                                batch_dims,
                                instances_per_row=points,
                                row_count=row_count,
                            )

                        case _:
                            raise NotImplementedError

                case rr.Scalars.columns:
                    match torch.atleast_1d(value := kwargs[key := "scalars"]):
                        case Tensor(shape=(*batch_dims, dim)):
                            lengths = partition_lengths(
                                batch_dims, instances_per_row=dim, row_count=row_count
                            )

                        case _:
                            logger.error("not implemented", key=key, value=value)
                            raise NotImplementedError
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
        time_column_cache: dict[TimeIndexCacheKey, TimeColumnBundle] = {}

        for path, component_column_configs in self._schema.component_columns.items():
            for column_config in component_column_configs:
                cache_key = _time_index_cache_key(column_config.time_index)
                if cache_key not in time_column_cache:
                    time_column_cache[cache_key] = self._build_time_columns(
                        data, column_config.time_index
                    )

                time_column_bundle = time_column_cache[cache_key]
                component_columns = self._build_component_columns(
                    column_config, data, row_count=time_column_bundle.row_count
                )

                rr.send_columns(
                    entity_path=self._entity_path(path),
                    indexes=time_column_bundle.columns,
                    columns=component_columns,
                )


def atleast_nd_left[T](x: torch.Tensor, n: int) -> torch.Tensor:
    return x.reshape((1,) * max(0, n - x.ndim) + tuple(x.shape))


def partition_lengths(
    batch_dims: Sequence[int], *, instances_per_row: int, row_count: int | None
) -> list[int]:
    if row_count is None:
        return [instances_per_row] * prod(batch_dims)

    total_instances = prod(batch_dims) * instances_per_row
    if row_count == 0:
        if total_instances == 0:
            return []

        msg = (
            "component instance count must be divisible by time row count: "
            f"{total_instances=} {row_count=}"
        )
        raise ValueError(msg)

    if total_instances % row_count != 0:
        msg = (
            "component instance count must be divisible by time row count: "
            f"{total_instances=} {row_count=}"
        )
        raise ValueError(msg)

    return [total_instances // row_count] * row_count
