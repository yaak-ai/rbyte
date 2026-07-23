from os import PathLike
from typing import Annotated, Any, final

import polars as pl
from m2df import Converter, MessageType
from polars.datatypes import DataType
from pydantic import BeforeValidator, InstanceOf, validate_call
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from xxhash import xxh3_64_hexdigest as digest

logger = get_logger(__name__)


@final
class YaakMetadataDataFrameBuilder:
    __name__ = __qualname__

    @validate_call
    def __init__(
        self,
        *,
        messages: dict[
            Annotated[
                MessageType,
                BeforeValidator(
                    lambda msg: (
                        getattr(MessageType, msg) if isinstance(msg, str) else msg
                    )
                ),
            ],
            dict[str, InstanceOf[DataType] | type[DataType] | None] | None,
        ],
    ) -> None:
        super().__init__()

        self._messages = messages
        self._converter = Converter(messages=self._messages)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_converter")
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._converter = Converter(messages=self._messages)

    def __pipefunc_hash__(self) -> str:  # ruff:ignore[bad-dunder-method-name]
        return digest(str(self._messages))

    def __call__(self, path: PathLike[str]) -> dict[str, pl.DataFrame]:
        with bound_contextvars(path=path):
            result = self._build(path)
            logger.debug(
                "built dataframes", length={k: len(v) for k, v in result.items()}
            )

            return result

    def _build(self, path: PathLike[str]) -> dict[str, pl.DataFrame]:
        dfs = {
            message_type.name: df
            for message_type, df in self._converter.convert(path).items()
        }

        if (df := dfs.pop((k := MessageType.ImageMetadata.name), None)) is not None:
            dfs |= {
                ".".join((k, *map(str, k_partition))): df_partition
                for k_partition, df_partition in df.partition_by(
                    "camera_name", include_key=False, as_dict=True
                ).items()
            }

        return dfs
