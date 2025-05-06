from ._duckdb import DuckDbDataFrameBuilder
from ._numpy import NumpyTensorSource
from .dataframe import (
    DataFrameAligner,
    DataFrameConcater,
    DataFrameDuckDbQuery,
    DataFrameGroupByDynamic,
    DataFrameIndexer,
)
from .path import PathDataFrameBuilder, PathTensorSource

__all__: list[str] = [
    "DataFrameAligner",
    "DataFrameConcater",
    "DataFrameDuckDbQuery",
    "DataFrameGroupByDynamic",
    "DataFrameIndexer",
    "DuckDbDataFrameBuilder",
    "NumpyTensorSource",
    "PathDataFrameBuilder",
    "PathTensorSource",
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
except ImportError:
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
    from .geo import WaypointBuilder, WaypointNormalizer
except ImportError:
    pass
else:
    __all__ += ["WaypointBuilder", "WaypointNormalizer"]
