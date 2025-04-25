from collections.abc import Sequence
from enum import StrEnum, auto, unique
from os import PathLike
from typing import cast, final

import more_itertools as mit
import polars as pl
import rerun.dataframe as rrd
from pydantic import validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars

logger = get_logger(__name__)


@unique
class Column(StrEnum):
    log_tick = auto()
    log_time = auto()


@final
class RrdDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(self, index: str, contents: dict[str, Sequence[str] | None]) -> None:
        self._index = index
        self._contents = contents

    def __call__(self, path: PathLike[str]) -> dict[str, pl.DataFrame]:
        with bound_contextvars(path=path):
            result = self._build(path)
            logger.debug(
                "built dataframes", length={k: len(v) for k, v in result.items()}
            )

            return result

    def _build(self, path: PathLike[str]) -> dict[str, pl.DataFrame]:
        recording = rrd.load_recording(path)
        schema = recording.schema()

        # Entity contents must include a non-static component to get index values.
        extra_contents: dict[str, Sequence[str]] = {}
        for entity_path, components in self._contents.items():
            if components is None or all(
                (col := schema.column_for(entity_path, component)) is not None
                and col.is_static
                for component in components
            ):
                non_static_components = [
                    col.component_name.removeprefix("rerun.components.")
                    for col in schema.component_columns()
                    if not col.is_static and col.entity_path == entity_path
                ]

                extra_component = mit.first(
                    (comp for comp in non_static_components if "Indicator" in comp),
                    default=mit.first(non_static_components),
                )

                extra_contents[entity_path] = [extra_component]

        view = recording.view(
            index=self._index,
            contents={
                entity_path: [*(components or []), *extra_contents.get(entity_path, [])]
                for entity_path, components in self._contents.items()
            },
            include_indicator_columns=True,
        )

        recording_df = cast(pl.DataFrame, pl.from_arrow(view.select().read_all())).drop(  # pyright: ignore[reportUnknownMemberType]
            Column.log_tick, Column.log_time
        )

        entity_columns = mit.map_reduce(
            recording_df.select(pl.exclude(self._index)).columns,
            keyfunc=lambda x: x.split(":")[0],
        )

        return {
            entity: recording_df.select(
                self._index, pl.col(*columns).name.map(lambda x: x.split(":")[1])
            )
            .drop_nulls()
            .drop(extra_contents.get(entity, []))
            for entity, columns in entity_columns.items()
        }
