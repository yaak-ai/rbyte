from ._json import JsonDataFrameBuilder
from ._numpy import NumpyTensorSource
from .dataframe import (
    DataFrameAligner,
    DataFrameConcater,
    DataFrameFilter,
    DataFrameFpsResampler,
    DataFrameIndexer,
)
from .path import PathDataFrameBuilder, PathTensorSource

__all__: list[str] = [
    "DataFrameAligner",
    "DataFrameConcater",
    "DataFrameFilter",
    "DataFrameFpsResampler",
    "DataFrameIndexer",
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
    from .yaak import YaakMetadataDataFrameBuilder, build_yaak_metadata_dataframe
except ImportError:
    pass
else:
    __all__ += ["YaakMetadataDataFrameBuilder", "build_yaak_metadata_dataframe"]
