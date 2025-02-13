from collections.abc import Iterable, Mapping, Sequence
from functools import cache, cached_property
from math import prod
from typing import Annotated, Any, Protocol, Self, cast, override, runtime_checkable

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb
from pydantic import (
    BeforeValidator,
    Field,
    ImportString,
    RootModel,
    model_validator,
    validate_call,
)
from pydantic.types import AnyType
from rerun._baseclasses import (
    Archetype,  # noqa: PLC2701
    ComponentColumn,
)
from rerun._send_columns import TimeColumnLike as _TimeColumnLike  # noqa: PLC2701
from structlog import get_logger
from structlog.contextvars import bound_contextvars

from rbyte.batch import Batch
from rbyte.config import BaseModel

from .base import Logger

logger = get_logger(__name__)


@runtime_checkable
class TimeColumnLike(_TimeColumnLike, Protocol): ...


class ImageFormat(BaseModel):
    pixel_format: (
        Annotated[rr.PixelFormat, BeforeValidator(rr.PixelFormat.auto)] | None
    ) = None

    color_model: (
        Annotated[rr.ColorModel, BeforeValidator(rr.ColorModel.auto)] | None
    ) = None

    @model_validator(mode="after")
    def validate_model(self: Self) -> Self:
        if not (bool(self.pixel_format) ^ bool(self.color_model)):
            msg = "pixel_format xor color_model must be specified"
            raise ValueError(msg)

        return self


RerunImportString = Annotated[
    ImportString[AnyType],
    BeforeValidator(
        lambda x: f"rerun.{x}"
        if isinstance(x, str) and not x.startswith("rerun.")
        else x
    ),
]


TimeConfig = RerunImportString[type[TimeColumnLike]]

ComponentConfig = (
    RerunImportString[type[Archetype]]
    | Annotated[
        Mapping[RerunImportString[type[rr.Image | rr.DepthImage]], ImageFormat],
        Field(max_length=1),
    ]
)


class Schema(RootModel[Mapping[str, TimeConfig | ComponentConfig]]):
    @cached_property
    def indexes(self) -> Mapping[str, TimeColumnLike]:
        return {k: v for k, v in self.root.items() if isinstance(v, TimeColumnLike)}

    @cached_property
    def columns(
        self,
    ) -> Mapping[
        str, type[Archetype] | Mapping[type[rr.Image | rr.DepthImage], ImageFormat]
    ]:
        return {k: v for k, v in self.root.items() if not isinstance(v, TimeColumnLike)}  # pyright: ignore[reportReturnType]


class RerunLogger(Logger[Batch]):
    @validate_call(config=BaseModel.model_config)
    def __init__(
        self,
        *,
        schema: Schema,
        spawn: bool = True,
        blueprint: rrb.BlueprintLike | None = None,
    ) -> None:
        super().__init__()

        self._schema: Schema = schema
        self._spawn: bool = spawn
        self._blueprint: rrb.BlueprintLike | None = blueprint

    @cache  # noqa: B019
    def _get_recording(self, application_id: str) -> rr.RecordingStream:
        recording = rr.new_recording(
            application_id, spawn=self._spawn, make_default=True
        )
        if self._blueprint is not None:
            rr.send_blueprint(self._blueprint, recording=recording)

        return recording

    @classmethod
    def _build_columns(
        cls,
        array: npt.NDArray[Any],
        schema: type[Archetype] | Mapping[type[rr.Image | rr.DepthImage], ImageFormat],
    ) -> Iterable[ComponentColumn]:
        match schema:
            case rr.Scalar:
                return rr.Scalar.columns(scalar=array)

            case rr.Points3D:
                match shape := array.shape:
                    case (3,):
                        return rr.Points3D.columns(positions=array)

                    case (*batch_dims, n, 3):
                        return rr.Points3D.columns(
                            positions=array.reshape(-1, 3)
                        ).partition([n] * prod(batch_dims))

                    case _:
                        logger.debug("not implemented", shape=shape)

                        raise NotImplementedError

            case rr.Tensor:
                return rr.Tensor.columns(data=array)

            case {rr.Image: image_format} | {rr.DepthImage: image_format}:
                with bound_contextvars(image_format=image_format, shape=array.shape):
                    match (
                        image_format.pixel_format,
                        image_format.color_model,
                        array.shape,
                    ):
                        case None, rr.ColorModel(), (*batch_dims, height, width, _):
                            pass

                        case rr.PixelFormat.NV12, None, (*batch_dims, dim, width):
                            height = int(dim / 1.5)

                        case _:
                            logger.error("not implemented")

                            raise NotImplementedError

                format = rr.components.ImageFormat(
                    height=height,
                    width=width,
                    pixel_format=image_format.pixel_format,
                    color_model=image_format.color_model,
                    channel_datatype=rr.ChannelDatatype.from_np_dtype(array.dtype),
                )

                batch_dim = prod(batch_dims)

                return rr.Image.columns(
                    buffer=array.reshape(batch_dim, -1).view(np.uint8),
                    format=[format] * batch_dim,
                )

            case _:
                logger.error("not implemented")

                raise NotImplementedError

    @override
    def log(self, batch_idx: int, batch: Batch) -> None:
        for i, sample in enumerate(batch.data):  # pyright: ignore[reportArgumentType, reportUnknownVariableType]
            with self._get_recording(batch.meta.input_id[i]):  # pyright: ignore[reportUnknownArgumentType, reportOptionalSubscript, reportUnknownMemberType, reportOptionalMemberAccess]
                indexes: Sequence[TimeColumnLike] = [
                    index(
                        timeline=timeline,
                        times=np.atleast_1d(sample.get(timeline).numpy()),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportCallIssue]
                    )
                    for timeline, index in self._schema.indexes.items()
                ]

                for entity_path, schema in self._schema.columns.items():
                    with bound_contextvars(path=entity_path, schema=schema):
                        array = cast(
                            npt.NDArray[Any],
                            sample.get(entity_path).cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
                        )

                        columns = self._build_columns(array, schema)
                        rr.send_columns(entity_path, indexes, columns, strict=True)
