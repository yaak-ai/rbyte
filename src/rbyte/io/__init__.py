from ._duckdb import DuckDBDataFrameQuery
from ._numpy import NumpyTensorSource
from .dataframe import (
    DataFrameAligner,
    DataFrameConcater,
    DataFrameGroupByDynamic,
    DataFrameIndexer,
)
from .path import PathDataFrameBuilder, PathTensorSource
from .tree import TreeBroadcastMapper, TreeItemGetter

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
    "TreeItemGetter",
]


try:  # ruff:ignore[non-empty-init-module]
    from .hdf5 import Hdf5DataFrameBuilder, Hdf5TensorSource
except ImportError:
    pass
else:
    __all__ += ["Hdf5DataFrameBuilder", "Hdf5TensorSource"]

try:  # ruff:ignore[non-empty-init-module]
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

try:  # ruff:ignore[non-empty-init-module]
    from .video import TorchCodecFrameSource, VideoDataFrameBuilder
except (ImportError, RuntimeError):
    pass
else:
    __all__ += ["TorchCodecFrameSource", "VideoDataFrameBuilder"]

try:  # ruff:ignore[non-empty-init-module]
    from .yaak import RouteMatchedWaypointGenerator, YaakMetadataDataFrameBuilder
except ImportError:
    pass
else:
    __all__ += ["RouteMatchedWaypointGenerator", "YaakMetadataDataFrameBuilder"]
