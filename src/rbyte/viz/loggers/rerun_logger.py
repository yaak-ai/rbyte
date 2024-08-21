from collections.abc import Callable, Mapping
from functools import cache, cached_property
from typing import Any, Literal, cast, override

import rerun as rr
import torch
from pydantic import Field, ImportString
from rerun._baseclasses import ComponentBatchMixin  # noqa: PLC2701
from torch import Tensor
from torch.utils._pytree import tree_flatten_with_path  # noqa: PLC2701

from rbyte.batch import Batch
from rbyte.config.base import BaseModel

from .base import Logger

type NestedKey = str | tuple[str, ...]
TimeColumn = rr.TimeSequenceColumn | rr.TimeNanosColumn | rr.TimeSecondsColumn


class SchemaItemConfig(BaseModel):
    key: NestedKey
    type: Callable[[str, int], None] | Callable[[str, float], None]


class TransformConfig(BaseModel):
    select: tuple[NestedKey, ...]
    apply: Callable[[Tensor], Tensor]


class RerunLoggerConfig(BaseModel):
    log_schema: Mapping[Literal["frame", "table"], Mapping[str, ImportString[Any]]]
    transforms: list[TransformConfig] = Field(default_factory=list)

    @cached_property
    def times(self) -> tuple[tuple[tuple[str, ...], type[TimeColumn]], ...]:
        return tuple(  # pyright: ignore[reportUnknownVariableType]
            (tuple(x.key for x in path), leaf)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
            for path, leaf in tree_flatten_with_path(self.log_schema)[0]
            if issubclass(leaf, TimeColumn)
        )

    @cached_property
    def components(
        self,
    ) -> tuple[tuple[tuple[str, ...], type[ComponentBatchMixin]], ...]:
        return tuple(  # pyright: ignore[reportUnknownVariableType]
            (tuple(x.key for x in path), leaf)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
            for path, leaf in tree_flatten_with_path(self.log_schema)[0]
            if issubclass(leaf, ComponentBatchMixin)
        )


class RerunLogger(Logger[Batch]):
    def __init__(self, config: object) -> None:
        super().__init__()

        self.config = RerunLoggerConfig.model_validate(config)

    @cache  # noqa: B019
    def _get_recording(self, *, application_id: str) -> rr.RecordingStream:  # noqa: PLR6301
        return rr.new_recording(
            application_id=application_id, spawn=True, make_default=True
        )

    @override
    def log(self, batch_idx: int, batch: Batch) -> None:  # pyright: ignore[reportGeneralTypeIssues, reportUnknownParameterType]
        for transform in self.config.transforms:
            batch = batch.update(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                batch.select(*transform.select).apply(transform.apply)  # pyright: ignore[reportUnknownMemberType]
            )

        # NOTE: zip because batch.meta.input_id is NonTensorData and isn't indexed
        for input_id, sample in zip(  # pyright: ignore[reportUnknownVariableType]
            batch.get(k := ("meta", "input_id")),  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            batch.exclude(k),  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            strict=True,
        ):
            with self._get_recording(application_id=input_id):  # pyright: ignore[reportUnknownArgumentType]
                times = [
                    fn(times=sample.get(k).numpy(), timeline="/".join(k))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    for k, fn in self.config.times
                ]

                for k, fn in self.config.components:
                    path = "/".join(k)
                    tensor = cast(Tensor, sample.get(k))  # pyright: ignore[reportUnknownMemberType]
                    match fn:
                        case rr.components.ImageBufferBatch:
                            match tensor.shape, tensor.dtype:
                                case ((_, height, width, 3), torch.uint8):
                                    # TODO: make this configurable?  # noqa: FIX002
                                    rr.log(
                                        path,
                                        [
                                            rr.components.ImageFormat(
                                                height=height,
                                                width=width,
                                                color_model="RGB",
                                                channel_datatype="U8",
                                            ),
                                            rr.Image.indicator(),
                                        ],
                                        static=True,
                                        strict=True,
                                    )

                                    # https://github.com/rerun-io/rerun/blob/46a7035bca81f4ff158e0975a5a78746fc2c730c/docs/snippets/all/archetypes/image_send_columns.py#L26
                                    tensor = tensor.flatten(start_dim=1, end_dim=-1)

                                case _:
                                    raise NotImplementedError

                        case _:
                            pass

                    rr.send_columns(
                        entity_path="/".join(k),
                        times=times,
                        components=[fn(tensor.numpy())],  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportCallIssue]
                        strict=True,
                    )
