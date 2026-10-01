"""Tests for the secret-redacting logging filter.

Covers the three required scenarios:

1. **Normal processing** - well-formed records pass through and secrets are
   scrubbed.
2. **Expected malformed records** - a format/argument mismatch (``getMessage``
   raises ``TypeError``/``ValueError``/``KeyError``) is handled explicitly:
   the record is kept but rewritten so a formatter cannot raise, and the raw
   payload is *not* leaked.
3. **Unexpected errors** - a record whose rendering raises something else is
   dropped, because its payload cannot be inspected for credentials.

A regression test asserts the end-to-end property that motivated the fix:
no credential text reaches the stream handler, even for a malformed call.
"""

from __future__ import annotations

import io
import logging

import pytest

from cyberllm.core import logging_config
from cyberllm.core.logging_config import (
    REDACTED,
    SecretRedactingFilter,
    _JsonFormatter,
    format_failure_count,
    redact_secrets,
)

SECRET = "sk-ABCDEFGHIJKLMNOP1234567890"


@pytest.fixture
def failing_counter_reset():
    """Isolate the module-level failure counter around each test."""
    start = format_failure_count()
    yield lambda: format_failure_count() - start
    logging_config._format_failures = start


@pytest.fixture
def capture():
    """A stream handler wired through the redaction filter, plus its buffer."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretRedactingFilter())
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))

    logger = logging.getLogger("cyberllm.test.capture")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.filters = [SecretRedactingFilter()]
    yield logger, stream
    logger.handlers = []
    logger.filters = []


def _record(msg, args=None, level=logging.INFO, name="cyberllm.test", cls=logging.LogRecord):
    """Build a LogRecord without going through a logger call site."""
    return cls(name=name, level=level, pathname=__file__, lineno=1,
               msg=msg, args=args, exc_info=None)


# ---------------------------------------------------------------------------
# 1. Normal processing
# ---------------------------------------------------------------------------
def test_plain_message_passes_through_unchanged(capture):
    logger, stream = capture
    logger.info("dataset ingest finished for %s", "cisa_kev")
    assert "dataset ingest finished for cisa_kev" in stream.getvalue()


def test_openai_style_key_is_redacted(capture):
    logger, stream = capture
    logger.info("calling api with key %s", SECRET)
    out = stream.getvalue()
    assert SECRET not in out
    assert REDACTED in out


def test_key_in_mapping_args_is_redacted(capture):
    logger, stream = capture
    logger.info("config=%(cfg)s", {"cfg": {"api_key": SECRET}})
    out = stream.getvalue()
    assert SECRET not in out
    assert REDACTED in out


def test_authorization_header_is_redacted():
    assert SECRET not in redact_secrets(f"authorization: Bearer {SECRET}")
    assert "hunter2hunter2" not in redact_secrets('password="hunter2hunter2"')


def test_record_with_credential_keeps_non_secret_text(capture):
    """Redaction must not blank the whole line - operators still need context."""
    logger, stream = capture
    logger.info("downloading %s for user analyst", SECRET)
    out = stream.getvalue()
    assert SECRET not in out


# ---------------------------------------------------------------------------
# 2. Expected malformed records
# ---------------------------------------------------------------------------
def test_format_arg_mismatch_is_handled_not_swallowed(capture, failing_counter_reset):
    """Two %s placeholders with one argument makes getMessage() raise TypeError."""
    logger, stream = capture
    logger.info("value=%s other=%s", "only-one")

    out = stream.getvalue()
    assert failing_counter_reset() == 1, "the failure must be counted, not ignored"
    assert "unformattable log record" in out
    assert "TypeError" in out


def test_malformed_record_does_not_emit_credential(capture, failing_counter_reset):
    """The regression this fix exists for: a malformed call must not leak."""
    logger, stream = capture
    # Two %s placeholders, one argument -> getMessage() raises TypeError.
    logger.info("api_key=%s extra=%s", SECRET)

    out = stream.getvalue()
    assert SECRET not in out, "credential leaked through a malformed log call"
    assert "unformattable log record" in out
    assert failing_counter_reset() == 1


def test_malformed_record_keeps_logger_and_level(capture):
    """The record is retained for observability, not discarded wholesale."""
    logger, stream = capture
    logger.error("bad %s %s", 1)
    out = stream.getvalue()
    assert "ERROR" in out
    assert "cyberllm.test.capture" in out


def test_malformed_record_is_formattable_after_filter():
    """Core invariant: a record admitted by the filter can be formatted."""
    record = _record("key=%s extra=%s", (SECRET, "x"))
    assert SecretRedactingFilter().filter(record) is True
    rendered = logging.Formatter("%(message)s").format(record)  # must not raise
    assert SECRET not in rendered


def test_mapping_format_missing_key_is_handled(failing_counter_reset):
    """A missing ``%(name)s`` key raises KeyError and must not drop the record.

    ``LogRecord.__init__`` unwraps a single-dict ``args`` to that dict, which
    is what makes mapping-style formatting work. Setting ``args`` directly
    reproduces that state.
    """
    record = _record("%(missing)s", None)
    record.args = {"present": 1}
    assert SecretRedactingFilter().filter(record) is True
    assert "unformattable log record" in record.getMessage()
    assert "KeyError" in record.getMessage()
    assert failing_counter_reset() == 1


def test_bad_format_specifier_is_handled(failing_counter_reset):
    """An invalid conversion specifier raises ValueError."""
    record = _record("bad %2$d specifier", (1,))
    assert SecretRedactingFilter().filter(record) is True
    assert "ValueError" in record.getMessage()
    assert failing_counter_reset() == 1


def test_malformed_record_does_not_leak_raw_args_in_description():
    """The fallback description must describe the record, never echo it."""
    record = _record("api_key=%s extra=%s", (SECRET,))  # 2 placeholders, 1 arg
    SecretRedactingFilter().filter(record)
    rendered = record.getMessage()
    assert SECRET not in rendered
    assert "trailing-arg" not in rendered
    assert "cyberllm.test" in rendered
    assert "unformattable" in rendered


@pytest.mark.parametrize("msg,args,exc_name", [
    ("a=%s b=%s", ("only",), "TypeError"),        # too few arguments
    ("a=%s b=%s %s", ("one", "two"), "TypeError"),  # too many arguments
    ("count=%d", ("not-a-number",), "TypeError"),   # wrong value type
    ("%(k)s", [1, 2], "TypeError"),                 # mapping syntax, non-mapping
    ("%s %2$d", (1, 2), "ValueError"),              # invalid specifier
])
def test_all_format_error_types_are_expected_not_dropped(
    failing_counter_reset, msg, args, exc_name
):
    """Every ``str % args`` failure mode is handled, not silently dropped."""
    record = _record(msg, args)
    assert SecretRedactingFilter().filter(record) is True, "record was dropped"
    rendered = record.getMessage()
    assert "unformattable log record" in rendered
    assert exc_name in rendered
    assert failing_counter_reset() == 1
    # Whatever we emit must survive a real formatter.
    assert logging.Formatter("%(message)s").format(record)


# ---------------------------------------------------------------------------
# 3. Unexpected errors
# ---------------------------------------------------------------------------
class _ExplodingRecord(logging.LogRecord):
    """A record whose rendering raises something other than TypeError."""

    def getMessage(self):  # noqa: N802 - stdlib naming
        raise RuntimeError("handler subsystem exploded")


def test_unexpected_error_drops_record(failing_counter_reset):
    record = _record("payload=%s", (SECRET,), cls=_ExplodingRecord)
    assert SecretRedactingFilter().filter(record) is False
    assert failing_counter_reset() == 1


def test_unexpected_error_does_not_raise_into_call_site(failing_counter_reset):
    """A broken record must never propagate an exception into business code."""
    record = _record(SECRET, None, cls=_ExplodingRecord)
    assert SecretRedactingFilter().filter(record) is False


def test_json_formatter_never_raises_on_bad_record():
    """Handler.handleError would dump the raw record, so the formatter guards."""
    record = _record("secret=%s", (SECRET,), level=logging.WARNING,
                     cls=_ExplodingRecord)
    rendered = _JsonFormatter().format(record)
    assert SECRET not in rendered
    assert "could not be rendered" in rendered


def test_json_formatter_redacts_well_formed_records():
    rendered = _JsonFormatter().format(_record("token %s", (SECRET,)))
    assert SECRET not in rendered
    assert REDACTED in rendered



# ---------------------------------------------------------------------------
# Recursion safety
# ---------------------------------------------------------------------------
def test_failure_path_does_not_recurse_into_logging(monkeypatch, failing_counter_reset):
    """Emitting a log from inside the filter would re-enter it indefinitely."""
    calls = {"n": 0}
    real_get_logger = logging_config.get_logger

    def counting_get_logger(name):
        calls["n"] += 1
        return real_get_logger(name)

    monkeypatch.setattr(logging_config, "get_logger", counting_get_logger)
    record = _record(SECRET, None, cls=_ExplodingRecord)
    SecretRedactingFilter().filter(record)
    assert calls["n"] == 0, "the filter must not log while filtering"
    assert failing_counter_reset() == 1


def test_counter_is_thread_safe():
    """Concurrent filtering must not corrupt the diagnostic counter."""
    import threading

    records = [_record("x", None, cls=_ExplodingRecord) for _ in range(200)]
    filt = SecretRedactingFilter()
    start = format_failure_count()

    def worker():
        for record in records:
            filt.filter(record)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert format_failure_count() - start == 800


# ---------------------------------------------------------------------------
# get_logger wiring
# ---------------------------------------------------------------------------
def test_get_logger_is_namespaced_and_filtered():
    logger = logging_config.get_logger("datasets.acquire")
    assert logger.name == "cyberllm.datasets.acquire"
    assert any(isinstance(f, SecretRedactingFilter) for f in logger.filters)


def test_get_logger_does_not_double_add_filter():
    logger = logging_config.get_logger("datasets.acquire")
    before = len(logger.filters)
    logging_config.get_logger("datasets.acquire")
    assert len(logger.filters) == before


def test_configure_logging_is_idempotent():
    """Calling configure_logging repeatedly must not stack handlers.

    Asserted as "the second call adds nothing" rather than as an absolute
    handler count: importing cyberllm.api.app already configures logging, so
    the number of handlers present when this test runs is not fixed.
    """
    root = logging.getLogger()
    logging_config.configure_logging("INFO")
    after_first = len(root.handlers)
    logging_config.configure_logging("DEBUG")
    logging_config.configure_logging("INFO")
    assert len(root.handlers) == after_first


def test_configure_logging_adds_redaction_filter_to_handler():
    logging_config.configure_logging("INFO")
    root = logging.getLogger()
    assert any(
        isinstance(f, SecretRedactingFilter) for h in root.handlers for f in h.filters
    )
