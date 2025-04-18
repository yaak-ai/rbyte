from ._json import JsonDataFrameBuilder
from ._numpy import NumpyTensorSource
from .dataframe import (
    DataFrameAligner,
    DataFrameConcater,
    DataFrameFilter,
    DataFrameFpsResampler,
    DataFrameIndexer,
    DataFrameJoiner,
    DataFrameQuery,
    DataFrameWithColumns,
    FixedWindowSampleBuilder,
)
from .path import PathDataFrameBuilder, PathTensorSource

__all__: list[str] = [
    "DataFrameAligner",
    "DataFrameConcater",
    "DataFrameFilter",
    "DataFrameFpsResampler",
    "DataFrameIndexer",
    "DataFrameJoiner",
    "DataFrameQuery",
    "DataFrameWithColumns",
    "FixedWindowSampleBuilder",
    "JsonDataFrameBuilder",
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
    from .rrd import RrdDataFrameBuilder, RrdFrameSource
except ImportError:
    pass
else:
    __all__ += ["RrdDataFrameBuilder", "RrdFrameSource"]

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
    from .geo import GeoDataFrameBuilder, WaypointBuilder, WaypointNormalizer
except ImportError:
    pass
else:
    __all__ += ["GeoDataFrameBuilder", "WaypointBuilder", "WaypointNormalizer"]
