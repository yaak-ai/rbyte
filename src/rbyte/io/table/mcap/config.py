from collections.abc import Mapping

import polars as pl
from pydantic import ConfigDict, ImportString

from rbyte.config.base import BaseModel, HydraConfig

PolarsDataType = pl.DataType | pl.DataTypeClass


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields: Mapping[
        str,
        Mapping[str, HydraConfig[PolarsDataType] | ImportString[PolarsDataType] | None],
    ]

    validate_crcs: bool = False
