"""Logging configuration for CyberLLM.

Provides a single ``get_logger`` entry point and a ``SecretRedactingFilter``
that scrubs anything resembling an API key before it reaches a handler. This
is defence in depth: the platform never passes secrets to log calls, and the
filter catches accidental leaks too.

Failure handling in this module is deliberately *defensive rather than
permissive*. A record whose format string and arguments disagree makes
``LogRecord.getMessage()`` raise; if the filter then simply returned ``True``
the record would reach the handler, the formatter would raise the same error,
and ``logging.Handler.handleError`` would print the raw record - including an
unredacted credential - to stderr. The filter therefore always leaves the
record in a state a formatter can handle, and counts any such failure in
:func:`format_failure_count` for the health endpoint to surface.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from typing import Any

_CONFIGURED = False

#: Failures observed in :meth:`SecretRedactingFilter.filter`. Guarded by a lock
#: and never logged from inside the filter, because doing so would re-enter the
#: logging machinery we are currently inside of.
_format_failures = 0
_failures_lock = threading.Lock()

#: Patterns whose *values* must never appear in logs.
_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{12,})"),                 # OpenAI-style keys
    re.compile(r"(ghp_[A-Za-z0-9]{20,})"),                   # GitHub PATs
    re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)([^\s\"',}]{6,})"),
    re.compile(r"(?i)(authorization\"?\s*[:=]\s*\"?bearer\s+)([^\s\"',}]{6,})"),
    re.compile(r"(?i)(password\"?\s*[:=]\s*\"?)([^\s\"',}]{4,})"),
]

REDACTED = "***REDACTED***"


def redact_secrets(text: str) -> str:
    """Replace credential-looking substrings in ``text``.

    Safe to call with any string; used by the filter and by the JSON
    formatter so both redact identically.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda m: (m.group(1) + REDACTED) if m.lastindex and m.lastindex >= 2
            else REDACTED,
            text,
        )
    return text


def format_failure_count() -> int:
    """Number of records the redaction filter could not format normally.

    Exposed through the health endpoint so a misbehaving logging call site is
    observable without the logging system needing to report on itself.
    """
    with _failures_lock:
        return _format_failures


def _note_format_failure() -> None:
    """Increment the failure counter without touching the logging system."""
    global _format_failures
    with _failures_lock:
        _format_failures += 1


class SecretRedactingFilter(logging.Filter):
    """Replace credential-looking substrings with :data:`REDACTED`.

    The filter also guarantees that any record it admits can be rendered by a
    standard formatter. ``LogRecord.getMessage()`` performs ``msg % args`` and
    raises on a mismatch (e.g. two ``%s`` placeholders but one argument). In
    that case the record is rewritten to a redacted, argument-free string
    instead of being passed through to fail again downstream.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            # Format string and args disagree. This is a programming error in
            # the call site, not a logging-infrastructure failure, so the
            # record is kept (it carries level/logger/timestamp) but its body
            # is replaced by the redacted literal message.
            #
            # The four types above are the complete set of exceptions raised by
            # ``str % args`` formatting in CPython:
            #   TypeError  - placeholder/argument arity or type mismatch
            #   ValueError - invalid format specifier or bad numeric conversion
            #   KeyError   - ``%(name)s`` with no matching mapping key
            #   IndexError - positional index beyond the supplied sequence
            _note_format_failure()
            record.msg = redact_secrets(_describe_bad_record(record, exc))
            record.args = ()
            return True
        except Exception:
            # A LogRecord subclass (or a broken ``__str__``) raised something
            # unexpected. Dropping the record is the only safe option: we
            # cannot know whether its payload contains a credential. We do not
            # log from here - that would recurse through this same filter.
            _note_format_failure()
            return False

        cleaned = redact_secrets(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def _describe_bad_record(record: logging.LogRecord, exc: Exception) -> str:
    """Build a safe, secret-free description of an unformattable record.

    Only the logger name, level and exception *type* are included - never the
    raw ``msg``/``args``, which are exactly what may hold a credential.
    """
    return (
        f"<unformattable log record: {type(exc).__name__} while formatting "
        f"message from logger '{record.name}' at level {record.levelname}>"
    )


class _MaxLevelFilter(logging.Filter):
    """Drop records above a configured level (used to silence libs)."""

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.level


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Install the root handler. Idempotent - safe to call from tests."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if _CONFIGURED:
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        return

    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
    handler.addFilter(SecretRedactingFilter())
    handler.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root.addHandler(handler)

    # Third-party loggers are noisy and sometimes echo request payloads.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
                  "transformers", "filelock", "matplotlib", "openai"):
        logging.getLogger(noisy).addFilter(_MaxLevelFilter(logging.WARNING))
    _CONFIGURED = True


class _JsonFormatter(logging.Formatter):
    """Minimal structured formatter for log aggregation.

    ``getMessage()`` is called defensively: the redaction filter has already
    normalised well-formed records, but a handler can be attached without it,
    and a formatter must never raise - ``Handler.handleError`` would then dump
    the raw record to stderr.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json as _json

        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = "<unformattable log record>"
        except Exception:
            message = "<log record could not be rendered>"

        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(message),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return _json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``cyberllm.<name>``)."""
    full = name if name.startswith("cyberllm") else f"cyberllm.{name}"
    logger = logging.getLogger(full)
    if not any(isinstance(f, SecretRedactingFilter) for f in logger.filters):
        logger.addFilter(SecretRedactingFilter())
    return logger


__all__ = [
    "get_logger", "configure_logging", "SecretRedactingFilter", "REDACTED",
    "redact_secrets", "format_failure_count",
]
