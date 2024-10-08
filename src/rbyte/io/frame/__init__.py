from .directory import DirectoryFrameReader

__all__ = ["DirectoryFrameReader"]

try:
    from .video import VideoFrameReader
except ImportError:
    pass
else:
    __all__ += ["VideoFrameReader"]

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
