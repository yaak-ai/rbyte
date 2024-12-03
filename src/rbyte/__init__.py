from importlib.metadata import version

from .dataset import Dataset
from .sample import FixedWindowSampleBuilder

__version__ = version(__package__ or __name__)

__all__ = ["Dataset", "FixedWindowSampleBuilder", "__version__"]
