from ._json import JsonDataFrameBuilder
from ._numpy import NumpyTensorSource
from .dataframe import (
    DataFrameAligner,
    DataFrameConcater,
    DataFrameFilter,
    DataFrameFpsResampler,
    DataFrameIndexer,
    DataFrameJoiner,
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
    from ._mcap import McapDataFrameBuilder, McapTensorSource
except ImportError:
    pass
else:
    __all__ += ["McapDataFrameBuilder", "McapTensorSource"]

try:
    from .rrd import RrdDataFrameBuilder, RrdFrameSource
except ImportError:
    pass
else:
    __all__ += ["RrdDataFrameBuilder", "RrdFrameSource"]

try:
    from .video.ffmpeg_source import FfmpegFrameSource
except ImportError:
    pass
else:
    __all__ += ["FfmpegFrameSource"]

try:
    from .video.dataframe_builder import VideoDataFrameBuilder
except ImportError:
    pass
else:
    __all__ += ["VideoDataFrameBuilder"]

try:
    from .yaak.metadata import YaakMetadataDataFrameBuilder
except ImportError:
    pass
else:
    __all__ += ["YaakMetadataDataFrameBuilder"]

try:
    from .yaak.waypoints import YaakWaypointNormalizer, YaakWaypointPreprocessor
except ImportError:
    pass
else:
    __all__ += ["YaakWaypointNormalizer", "YaakWaypointPreprocessor"]
