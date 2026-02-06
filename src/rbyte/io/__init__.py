from ._duckdb import DuckDBDataFrameQuery
from ._numpy import NumpyTensorSource
from .dataframe import (
    DataFrameAligner,
    DataFrameConcater,
    DataFrameGroupByDynamic,
    DataFrameIndexer,
)
from .path import PathDataFrameBuilder, PathTensorSource
from .tree import TreeBroadcastMapper

__all__: list[str] = [
    "DataFrameAligner",
    "DataFrameConcater",
    "DataFrameGroupByDynamic",
    "DataFrameIndexer",
    "DuckDBDataFrameQuery",
    "NumpyTensorSource",
    "PathDataFrameBuilder",
    "PathTensorSource",
    "TreeBroadcastMapper",
]


try:  # noqa: RUF067
    from .hdf5 import Hdf5DataFrameBuilder, Hdf5TensorSource
except ImportError:
    pass
else:
    __all__ += ["Hdf5DataFrameBuilder", "Hdf5TensorSource"]

try:  # noqa: RUF067
    from ._mcap import (
        JsonMcapDecoderFactory,
        McapDataFrameBuilder,
        McapTensorSource,
        ProtobufMcapDecoderFactory,
    )
except ImportError:
    pass
else:
    __all__ += [
        "JsonMcapDecoderFactory",
        "McapDataFrameBuilder",
        "McapTensorSource",
        "ProtobufMcapDecoderFactory",
    ]

try:  # noqa: RUF067
    from .video import TorchCodecFrameSource, VideoDataFrameBuilder
except (ImportError, RuntimeError):
    pass
else:
    __all__ += ["TorchCodecFrameSource", "VideoDataFrameBuilder"]

try:  # noqa: RUF067
    from .yaak.metadata import YaakMetadataDataFrameBuilder
except ImportError:
    pass
else:
    __all__ += ["YaakMetadataDataFrameBuilder"]


try:  # noqa: RUF067
    from .geo import WaypointBuilder
except ImportError:
    pass
else:
    __all__ += ["WaypointBuilder"]
