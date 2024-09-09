from importlib.metadata import version

from .dataset import Dataset

__version__ = version(__package__ or __name__)

__all__ = ["Dataset", "__version__"]
