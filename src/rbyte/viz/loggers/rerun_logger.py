from collections.abc import Iterable, Mapping, Sequence
from functools import cache, cached_property
from typing import Annotated, Any, Protocol, Self, cast, override, runtime_checkable

import more_itertools as mit
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
    ComponentBatchLike,
)
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


TimeConfig = RerunImportString[type[TimeColumn]]

ComponentConfig = (
    RerunImportString[type[Archetype]]
    | Annotated[
        Mapping[RerunImportString[type[rr.Image | rr.DepthImage]], ImageFormat],
        Field(max_length=1),
    ]
)


class Schema(RootModel[Mapping[str, TimeConfig | ComponentConfig]]):
    @cached_property
    def times(self) -> Mapping[str, TimeColumn]:
        return {k: v for k, v in self.root.items() if isinstance(v, TimeColumn)}

    @cached_property
    def components(
        self,
    ) -> Mapping[
        str, type[Archetype] | Mapping[type[rr.Image | rr.DepthImage], ImageFormat]
    ]:
        return {k: v for k, v in self.root.items() if not isinstance(v, TimeColumn)}  # pyright: ignore[reportReturnType]


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
    def _build_components(
        cls,
        array: npt.NDArray[Any],
        schema: type[Archetype] | Mapping[type[rr.Image | rr.DepthImage], ImageFormat],
    ) -> Iterable[ComponentBatchLike]:
        match schema:
            case rr.Scalar:
                return [schema.indicator(), rr.components.ScalarBatch(array)]

            case rr.Points3D:
                match shape := array.shape:
                    case (n, 3):
                        batch = rr.components.Position3DBatch(array)

                    case (s, n, 3):
                        batch = rr.components.Position3DBatch(
                            array.reshape(-1, 3)
                        ).partition([n] * s)

                    case _:
                        logger.debug("not implemented", shape=shape)

                        raise NotImplementedError

                return [schema.indicator(), batch]

            case rr.Tensor:
                return [schema.indicator(), rr.components.TensorDataBatch(array)]

            case {rr.Image: image_format} | {rr.DepthImage: image_format}:
                with bound_contextvars(image_format=image_format, shape=array.shape):
                    match (
                        image_format.pixel_format,
                        image_format.color_model,
                        array.shape,
                    ):
                        case None, rr.ColorModel(), (_batch, height, width, _):
                            pass

                        case rr.PixelFormat.NV12, None, (_batch, dim, width):
                            height = int(dim / 1.5)

                        case _:
                            logger.error("not implemented")

                            raise NotImplementedError

                image_format = rr.components.ImageFormat(
                    height=height,
                    width=width,
                    pixel_format=image_format.pixel_format,
                    color_model=image_format.color_model,
                    channel_datatype=rr.ChannelDatatype.from_np_dtype(array.dtype),
                )
                return [
                    mit.one(schema).indicator(),
                    rr.components.ImageFormatBatch([image_format] * _batch),
                    rr.components.ImageBufferBatch(
                        array.reshape(_batch, -1).view(np.uint8)
                    ),
                ]

            case _:
                logger.error("not implemented")

                raise NotImplementedError

    @override
    def log(self, batch_idx: int, batch: Batch) -> None:
        for i, sample in enumerate(batch.data):  # pyright: ignore[reportUnknownVariableType]
            with self._get_recording(batch.meta.input_id[i]):  # pyright: ignore[reportUnknownArgumentType, reportIndexIssue]
                times: Sequence[TimeColumn] = [
                    column(timeline=timeline, times=sample.get(timeline).numpy())  # pyright: ignore[reportUnknownMemberType, reportCallIssue]
                    for timeline, column in self._schema.times.items()
                ]

                for entity_path, schema in self._schema.components.items():
                    with bound_contextvars(path=entity_path, schema=schema):
                        array = cast(
                            npt.NDArray[Any],
                            sample.get(entity_path).cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
                        )

                        components = self._build_components(array, schema)
                        rr.send_columns(
                            entity_path=entity_path,
                            times=times,
                            components=components,  # pyright: ignore[reportArgumentType]
                            strict=True,
                        )
