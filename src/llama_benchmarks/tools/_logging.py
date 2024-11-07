from contextlib import contextmanager
import logging
from logging import Logger
from time import perf_counter_ns as timer

from ._common import default_arg

__all__ = [
    "trace",
]


@contextmanager
def trace(logger: Logger, prefix: str, log_level: int | None = None):
    # Defaults
    log_level = default_arg(log_level, logging.DEBUG)

    start_time = timer()

    try:
        yield
    finally:
        duration = (timer() - start_time) / 1000000
        logger.log(log_level, f"{prefix} took {duration:0.0f} ms")
