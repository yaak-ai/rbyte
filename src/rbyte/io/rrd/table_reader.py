import json
from collections.abc import Hashable, Mapping, Sequence
from enum import StrEnum, auto, unique
from os import PathLike
from typing import cast, override

import more_itertools as mit
import polars as pl
import rerun.dataframe as rrd
from optree import PyTree
from xxhash import xxh3_64_intdigest as digest

from rbyte.config.base import BaseModel
from rbyte.io.table.base import TableReader


class Config(BaseModel):
    index: str
    contents: Mapping[str, Sequence[str]]


@unique
class Column(StrEnum):
    log_tick = auto()
    log_time = auto()
    idx = "_idx_"


class RrdTableReader(TableReader, Hashable):
    def __init__(self, **kwargs: object) -> None:
        self._config: Config = Config.model_validate(kwargs)

    @override
    def read(self, path: PathLike[str]) -> PyTree[pl.DataFrame]:
        recording = rrd.load_recording(path)  # pyright: ignore[reportUnknownMemberType]
        schema = recording.schema()

        # Entity contents must include a non-static component to get index values.
        extra_contents: Mapping[str, Sequence[str]] = {}
        for entity_path, components in self._config.contents.items():
            match components:
                case [Column.idx, *rest] if all(
                    (col := schema.column_for(entity_path, component)) is not None
                    and col.is_static
                    for component in rest
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

                case _:
                    pass

        view = recording.view(
            index=self._config.index,
            contents={
                entity_path: [*components, *extra_contents.get(entity_path, [])]
                for entity_path, components in self._config.contents.items()
            },
            include_indicator_columns=True,
        )

        recording_df = cast(pl.DataFrame, pl.from_arrow(view.select().read_all())).drop(  # pyright: ignore[reportUnknownMemberType]
            Column.log_tick, Column.log_time
        )

        entity_columns = mit.map_reduce(
            recording_df.select(pl.exclude(self._config.index)).columns,
            keyfunc=lambda x: x.split(":")[0],
        )

        dfs: Mapping[str, pl.DataFrame] = {}

        for entity, columns in entity_columns.items():
            entity_df = (
                recording_df.select(
                    self._config.index,
                    pl.col(*columns).name.map(lambda x: x.split(":")[1]),
                )
                .drop_nulls()
                .drop(extra_contents.get(entity, []))
            )

            if Column.idx in self._config.contents[entity]:
                entity_df = entity_df.with_row_index(Column.idx)

            dfs[entity] = entity_df

        return dfs  # pyright: ignore[reportReturnType]

    @override
    def __hash__(self) -> int:
        config = self._config.model_dump_json()
        # roundtripping json to work around https://github.com/pydantic/pydantic/issues/7424
        config_str = json.dumps(json.loads(config), sort_keys=True)

        return digest(config_str)
