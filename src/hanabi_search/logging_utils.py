"""Logging setup for entry points."""

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Logs to stderr and, if given, to `log_file`.

    Args:
        level: Minimum level to log.
        log_file: Optional file that receives the same records.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=True)
