from typing import Generic, TypeVar

from structlog import get_logger
from typing_extensions import override

from .base import Logger

logger = get_logger(__name__)

T = TypeVar("T")


class ConsoleLogger(Logger[T], Generic[T]):
    @override
    def log(self, data: T) -> None:
        logger.info(None, data=data)
