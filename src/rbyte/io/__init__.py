from ._duckdb import DuckDBDataFrameQuery
from ._numpy import NumpyTensorSource, frombuffer
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
    "frombuffer",
]


try:
    from .hdf5 import Hdf5DataFrameBuilder, Hdf5TensorSource
except ImportError:
    pass
else:
    __all__ += ["Hdf5DataFrameBuilder", "Hdf5TensorSource"]

try:
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

try:
    from .video import TorchCodecFrameSource, VideoDataFrameBuilder
except (ImportError, RuntimeError):
    pass
else:
    __all__ += ["TorchCodecFrameSource", "VideoDataFrameBuilder"]

try:
    from .yaak.metadata import YaakMetadataDataFrameBuilder
except ImportError:
    pass
else:
    __all__ += ["YaakMetadataDataFrameBuilder"]


try:
    from .geo import WaypointBuilder
except ImportError:
    pass
else:
    __all__ += ["WaypointBuilder"]
