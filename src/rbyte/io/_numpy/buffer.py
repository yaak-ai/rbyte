import numpy as np


def frombuffer(
    buffer: np.ndarray,
    dtype: np.dtype,
    height: int = 600,
    width: int = 960,
    channels: int = 3,
) -> np.ndarray:

    return np.frombuffer(buffer, dtype=dtype).reshape([height, width, channels])
