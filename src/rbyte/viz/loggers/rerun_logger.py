from collections.abc import Mapping, Sequence
from functools import cache, cached_property
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    Self,
    cast,
    override,
    runtime_checkable,
)

import more_itertools as mit
import numpy as np
import numpy.typing as npt
import rerun as rr
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    ImportString,
    model_validator,
    validate_call,
)
from pydantic.types import AnyType
from rerun._baseclasses import Archetype  # noqa: PLC2701
from rerun._send_columns import TimeColumnLike  # noqa: PLC2701
from structlog import get_logger
from structlog.contextvars import bound_contextvars

from rbyte.batch import Batch
from rbyte.config import BaseModel

from .base import Logger

logger = get_logger(__name__)


@runtime_checkable
class TimeColumn(TimeColumnLike, Protocol): ...


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
            msg = "either pixel_format or color_model must be specified"
            raise ValueError(msg)

        return self


RerunImportString = Annotated[
    ImportString[AnyType],
    BeforeValidator(lambda x: f"rerun.{x}" if not x.startswith("rerun.") else x),
]

FrameConfig = Annotated[
    Mapping[RerunImportString[type[Archetype]], ImageFormat], Field(max_length=1)
]

TableConfig = RerunImportString[type[TimeColumn | Archetype]]


class Schema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame: Mapping[str, FrameConfig] = Field(default_factory=dict)
    table: Mapping[str, TableConfig] = Field(default_factory=dict)

    @cached_property
    def times(self) -> Mapping[tuple[Literal["table"], str], TimeColumn]:
        return {
            ("table", k): v for k, v in self.table.items() if isinstance(v, TimeColumn)
        }

    @cached_property
    def components(self) -> Mapping[tuple[str, str], FrameConfig | type[Archetype]]:
        return {("frame", k): v for k, v in self.frame.items()} | {
            ("table", k): v for k, v in self.table.items() if issubclass(v, Archetype)
        }


class RerunLogger(Logger[Batch]):
    @validate_call
    def __init__(self, schema: Schema) -> None:
        super().__init__()

        self._schema = schema

    @cache  # noqa: B019
    def _get_recording(self, *, application_id: str) -> rr.RecordingStream:  # noqa: PLR6301
        return rr.new_recording(
            application_id=application_id, spawn=True, make_default=True
        )

    @override
    def log(self, batch_idx: int, batch: Batch) -> None:
        # NOTE: zip because batch.meta.input_id is NonTensorData and isn't indexed
        for input_id, sample in zip(  # pyright: ignore[reportUnknownVariableType]
            batch.get(path := ("meta", "input_id")),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
            batch.exclude(path),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
            strict=True,
        ):
            with self._get_recording(application_id=input_id):  # pyright: ignore[reportUnknownArgumentType]
                times: Sequence[TimeColumn] = [
                    v(timeline="/".join(k), times=sample.get(k).numpy())  # pyright: ignore[reportUnknownMemberType, reportCallIssue]
                    for k, v in self._schema.times.items()
                ]

                for path, schema in self._schema.components.items():
                    with bound_contextvars(path=path, schema=schema):
                        arr = cast(npt.NDArray[Any], sample.get(path).cpu().numpy())  # pyright: ignore[reportUnknownMemberType]
                        match schema:
                            case rr.Scalar:
                                components = [
                                    schema.indicator(),
                                    rr.components.ScalarBatch(arr),
                                ]

                            case rr.Points3D:
                                components = [
                                    schema.indicator(),
                                    rr.components.Position3DBatch(arr).partition(
                                        arr.shape[0]
                                    ),
                                ]

                            case rr.Tensor:
                                components = [
                                    schema.indicator(),
                                    rr.components.TensorDataBatch(arr),
                                ]

                            case {rr.Image: image_format} | {
                                rr.DepthImage: image_format
                            }:
                                with bound_contextvars(
                                    image_format=image_format, shape=arr.shape
                                ):
                                    match (
                                        image_format.pixel_format,
                                        image_format.color_model,
                                        arr.shape,
                                    ):
                                        case None, rr.ColorModel(), (
                                            _batch,
                                            height,
                                            width,
                                            _,
                                        ):
                                            pass

                                        case rr.PixelFormat.NV12, None, (
                                            _batch,
                                            dim,
                                            width,
                                        ):
                                            height = int(dim / 1.5)

                                        case _:
                                            logger.error("not implemented")

                                            raise NotImplementedError

                                image_format = rr.components.ImageFormat(
                                    height=height,
                                    width=width,
                                    pixel_format=image_format.pixel_format,
                                    color_model=image_format.color_model,
                                    channel_datatype=rr.ChannelDatatype.from_np_dtype(
                                        arr.dtype
                                    ),
                                )
                                components = [
                                    mit.one(schema).indicator(),
                                    rr.components.ImageFormatBatch(
                                        [image_format] * _batch
                                    ),
                                    rr.components.ImageBufferBatch(
                                        arr.reshape(_batch, -1).view(np.uint8)
                                    ),
                                ]

                            case _:
                                logger.error("not implemented")

                                raise NotImplementedError

                        rr.send_columns(
                            entity_path="/".join(path),
                            times=times,
                            components=components,  # pyright: ignore[reportArgumentType]
                            strict=True,
                        )
