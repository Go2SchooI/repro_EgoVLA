import logging as std_logging
import typing

if typing.TYPE_CHECKING:
    from loguru import Logger
else:
    Logger = None

__all__ = ["logger"]


def __get_logger() -> Logger:
    try:
        from loguru import logger
        return logger
    except ImportError:
        logger = std_logging.getLogger("llava")
        if not logger.handlers:
            std_logging.basicConfig(level=std_logging.INFO)
        return logger


logger = __get_logger()
