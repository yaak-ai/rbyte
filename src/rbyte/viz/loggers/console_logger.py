from typing import TypeVar, override

from structlog import get_logger

from .base import Logger

logger = get_logger(__name__)

T = TypeVar("T")


class ConsoleLogger[T](Logger[T]):
    @override
    def log(self, data: T) -> None:
        logger.info(None, data=data)
