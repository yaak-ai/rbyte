from .directory import DirectoryFrameReader

__all__ = ["DirectoryFrameReader"]


try:
    from .mcap import McapFrameReader
except ImportError:
    pass
else:
    __all__ += ["McapFrameReader"]

try:
    from .hdf5 import Hdf5FrameReader
except ImportError:
    pass
else:
    __all__ += ["Hdf5FrameReader"]

try:
    from .rrd import RrdFrameReader
except ImportError:
    pass
else:
    __all__ += ["RrdFrameReader"]

try:
    from .video.ffmpeg_reader import FfmpegFrameReader
except ImportError:
    pass
else:
    __all__ += ["FfmpegFrameReader"]
