# app/core/logging.py
from __future__ import annotations

import logging
import os
from logging.config import dictConfig


def setup_logging() -> None:
    """
    Configure app + uvicorn logging with a consistent format.

    Control levels via env vars:
      - LOG_LEVEL (default: INFO)
      - UVICORN_LOG_LEVEL (default: INFO)
    """
    log_level = (os.getenv("LOG_LEVEL") or "INFO").upper()
    uvicorn_level = (os.getenv("UVICORN_LOG_LEVEL") or log_level).upper()

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "loggers": {
                # Your application logs
                "app": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False,
                },
                # Uvicorn logs
                "uvicorn": {
                    "handlers": ["console"],
                    "level": uvicorn_level,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["console"],
                    "level": uvicorn_level,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console"],
                    "level": uvicorn_level,
                    "propagate": False,
                },
            },
            "root": {
                "handlers": ["console"],
                "level": log_level,
            },
        }
    )

    # Make sure any "print-ish" libs don’t drown you
    logging.getLogger("sqlalchemy.engine").setLevel(
        os.getenv("SQLALCHEMY_LOG_LEVEL", "WARNING")
    )
