from ._json import JsonTableReader
from ._numpy import NumpyTensorSource
from .path import PathTableReader, PathTensorSource
from .table import FpsResampler, TableAligner, TableBuilder, TableConcater

__all__: list[str] = [
    "FpsResampler",
    "JsonTableReader",
    "NumpyTensorSource",
    "PathTableReader",
    "PathTensorSource",
    "TableAligner",
    "TableBuilder",
    "TableConcater",
]

try:
    from .hdf5 import Hdf5TableReader, Hdf5TensorSource
except ImportError:
    pass
else:
    __all__ += ["Hdf5TableReader", "Hdf5TensorSource"]

try:
    from ._mcap import McapTableReader, McapTensorSource
except ImportError:
    pass
else:
    __all__ += ["McapTableReader", "McapTensorSource"]

try:
    from .rrd import RrdFrameSource, RrdTableReader
except ImportError:
    pass
else:
    __all__ += ["RrdFrameSource", "RrdTableReader"]

try:
    from .video.ffmpeg_source import FfmpegFrameSource
except ImportError:
    pass
else:
    __all__ += ["FfmpegFrameSource"]

try:
    from .yaak import YaakMetadataTableReader
except ImportError:
    pass
else:
    __all__ += ["YaakMetadataTableReader"]
