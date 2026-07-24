from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_file: str | Path,
    *,
    level: str = "INFO",
    max_bytes: int = 1_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Route every "botchan.*" logger to a rotating file.

    The Rich dashboard owns the terminal, so nothing is written to
    stdout/stderr; the file is the persistent record of post attempts
    and errors.
    """
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )

    logger = logging.getLogger("botchan")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
