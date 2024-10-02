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

import numpy.typing as npt
import rerun as rr
from pydantic import (
    BeforeValidator,
    ConfigDict,
    ImportString,
    model_validator,
    validate_call,
)
from rerun._baseclasses import ComponentBatchMixin  # noqa: PLC2701
from rerun._send_columns import TimeColumnLike  # noqa: PLC2701

from rbyte.batch import Batch
from rbyte.config import BaseModel

from .base import Logger


@runtime_checkable
class TimeColumn(TimeColumnLike, Protocol): ...


class ImageFormatConfig(BaseModel):
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


TableSchema = (
    ImportString[type[TimeColumn]] | ImportString[type[rr.components.ScalarBatch]]
)
FrameSchema = Mapping[
    ImportString[type[rr.components.ImageBufferBatch]], ImageFormatConfig
]


class Schema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame: Mapping[str, FrameSchema]
    table: Mapping[str, TableSchema]

    @cached_property
    def times(self) -> Mapping[tuple[Literal["table"], str], TimeColumn]:
        return {
            ("table", k): v for k, v in self.table.items() if isinstance(v, TimeColumn)
        }

    @cached_property
    def components(
        self,
    ) -> Mapping[tuple[str, str], FrameSchema | type[ComponentBatchMixin]]:
        return {("frame", k): v for k, v in self.frame.items()} | {
            ("table", k): v
            for k, v in self.table.items()
            if issubclass(v, ComponentBatchMixin)
        }


class RerunLogger(Logger[Batch]):
    @validate_call
    def __init__(self, schema: Schema) -> None:
        super().__init__()

        self._schema = schema

    @cache  # noqa: B019
    def _get_recording(self, *, application_id: str) -> rr.RecordingStream:
        with rr.new_recording(
            application_id=application_id, spawn=True, make_default=True
        ) as recording:
            for k in self._schema.frame:
                rr.log(
                    entity_path=f"frame/{k}",
                    entity=[rr.Image.indicator()],
                    static=True,
                    strict=True,
                )

            return recording

    @override
    def log(self, batch_idx: int, batch: Batch) -> None:
        # NOTE: zip because batch.meta.input_id is NonTensorData and isn't indexed
        for input_id, sample in zip(  # pyright: ignore[reportUnknownVariableType]
            batch.get(k := ("meta", "input_id")),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
            batch.exclude(k),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
            strict=True,
        ):
            with self._get_recording(application_id=input_id):  # pyright: ignore[reportUnknownArgumentType]
                times: Sequence[TimeColumn] = [
                    v(timeline="/".join(k), times=sample.get(k).numpy())  # pyright: ignore[reportUnknownMemberType, reportCallIssue]
                    for k, v in self._schema.times.items()
                ]

                for k, v in self._schema.components.items():
                    tensor = cast(npt.NDArray[Any], sample.get(k).cpu().numpy())  # pyright: ignore[reportUnknownMemberType]
                    match v:
                        case rr.components.ScalarBatch:
                            components = [v(tensor)]

                        case {
                            rr.components.ImageBufferBatch: ImageFormatConfig(
                                pixel_format=pixel_format, color_model=color_model
                            )
                        }:
                            match (pixel_format, color_model, tensor.shape):
                                case None, rr.ColorModel.RGB, (
                                    (b, h, w, 3) | (b, 3, h, w)
                                ):
                                    image_format = rr.components.ImageFormat(
                                        width=w,
                                        height=h,
                                        color_model=color_model,
                                        channel_datatype=rr.ChannelDatatype.from_np_dtype(
                                            tensor.dtype
                                        ),
                                    )

                                case rr.PixelFormat.NV12, None, (b, dim, w):
                                    image_format = rr.components.ImageFormat(
                                        width=w,
                                        height=int(dim / 1.5),
                                        pixel_format=pixel_format,
                                    )

                                case _:
                                    raise NotImplementedError

                            components = [
                                rr.components.ImageFormatBatch([image_format] * b),
                                rr.components.ImageBufferBatch(tensor.reshape(b, -1)),
                            ]

                        case _:
                            raise NotImplementedError

                    rr.send_columns(
                        entity_path="/".join(k),
                        times=times,
                        components=components,
                        strict=True,
                    )
