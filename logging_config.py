"""Structured (JSON) logging setup for the service."""

import logging
import sys

from pythonjsonlogger import jsonlogger

from config import settings


def setup_logging() -> None:
    """Configure root logger with a JSON formatter."""
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    # Remove any pre-existing handlers to avoid duplicate logs.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Quieten noisy third-party loggers.
    for noisy in ("chromadb", "httpx", "httpcore", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
