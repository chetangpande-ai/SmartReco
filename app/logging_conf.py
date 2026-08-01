"""Structured logging with a request id threaded through every line.

The request id is also handed to the agent so a user-visible recommendation can be
traced back through ingest -> profile -> retrieval -> LLM call in one grep.
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar

from app.config import settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_BUILTIN = frozenset(
    "name msg args levelname levelno pathname filename module exc_info exc_text "
    "stack_info lineno funcName created msecs relativeCreated thread threadName "
    "processName process taskName message asctime".split()
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _BUILTIN}
        if extras:
            out.update(extras)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable for local dev; keeps the extras so they aren't lost."""

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<5} [{rid[:8]}] {record.name}: {record.getMessage()}"
        extras = {k: v for k, v in record.__dict__.items() if k not in _BUILTIN}
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # These are chatty and say nothing we don't already log ourselves.
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "apscheduler.executors"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def new_request_id() -> str:
    return uuid.uuid4().hex
