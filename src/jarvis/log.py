"""Structured logging setup (structlog) with per-conversation trace IDs.

Named ``log`` rather than ``logging`` to avoid shadowing the standard
library module in editors and tooling.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog

__all__ = ["configure_logging", "bind_trace_id", "clear_trace_id"]

LogFormat = Literal["console", "json"]


def configure_logging(level: str = "INFO", fmt: LogFormat = "console") -> None:
    """Configure structlog for the whole process.

    Logs go to stderr so the CLI's stdout stays clean for conversation
    text. ``json`` format emits one JSON object per line for ingestion;
    ``console`` is a human-friendly renderer for development.
    """
    renderer: structlog.typing.Processor
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def bind_trace_id(trace_id: str) -> None:
    """Attach ``trace_id`` to every log line in the current context."""
    structlog.contextvars.bind_contextvars(trace_id=trace_id)


def clear_trace_id() -> None:
    """Remove any bound trace id from the current context."""
    structlog.contextvars.unbind_contextvars("trace_id")
