__all__: list[str] = []

try:
    from .ffmpeg_reader import FfmpegFrameReader
except ImportError:
    pass

else:
    __all__ += ["FfmpegFrameReader"]

try:
    from .vali_reader import ValiGpuFrameReader
except ImportError:
    pass

else:
    __all__ += ["ValiGpuFrameReader"]
