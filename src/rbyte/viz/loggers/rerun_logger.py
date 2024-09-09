from collections.abc import Mapping
from functools import cache, cached_property
from typing import Any, Literal, Protocol, cast, override, runtime_checkable

import rerun as rr
import torch
from optree import tree_flatten_with_path
from pydantic import ImportString, validate_call
from rerun._baseclasses import ComponentBatchMixin  # noqa: PLC2701
from rerun._send_columns import TimeColumnLike  # noqa: PLC2701
from torch import Tensor

from rbyte.batch import Batch

from .base import Logger


@runtime_checkable
class TimeColumn(TimeColumnLike, Protocol): ...


class RerunLogger(Logger[Batch]):
    @validate_call
    def __init__(
        self,
        schema: Mapping[Literal["frame", "table"], Mapping[str, ImportString[Any]]],
    ) -> None:
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
            batch.get(k := ("meta", "input_id")),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
            batch.exclude(k),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
            strict=True,
        ):
            with self._get_recording(application_id=input_id):  # pyright: ignore[reportUnknownArgumentType]
                times = [
                    fn(timeline="/".join(k), times=sample.get(k).numpy())  # pyright: ignore[reportUnknownMemberType, reportCallIssue]
                    for k, fn in self.times
                ]

                for k, fn in self.components:
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

    @cached_property
    def times(self) -> tuple[tuple[tuple[str, ...], type[TimeColumn]], ...]:
        paths, leaves, _ = tree_flatten_with_path(self._schema)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]

        return tuple(
            (path, leaf)
            for path, leaf in zip(paths, leaves, strict=True)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            if issubclass(leaf, TimeColumn)
        )

    @cached_property
    def components(
        self,
    ) -> tuple[tuple[tuple[str, ...], type[ComponentBatchMixin]], ...]:
        paths, leaves, _ = tree_flatten_with_path(self._schema)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]

        return tuple(
            (path, leaf)
            for path, leaf in zip(paths, leaves, strict=True)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            if issubclass(leaf, ComponentBatchMixin)
        )
