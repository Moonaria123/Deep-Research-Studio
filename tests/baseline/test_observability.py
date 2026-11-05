"""Observability spans + logging are no-op safe without OTel (DR-ITER-024)."""

from __future__ import annotations

from deepresearch_core.observability import get_logger, span


def test_span_is_noop_without_otel():
    # No OTLP endpoint configured -> span must not raise and must run its body.
    ran = {"n": 0}
    with span("test.span", foo="bar", count=1):
        ran["n"] += 1
    assert ran["n"] == 1


def test_span_swallows_attribute_types():
    # Mixed attribute types should never crash the wrapped work.
    with span("test.span", s="x", i=2, f=1.5, b=True):
        result = 21 * 2
    assert result == 42


def test_get_logger_accepts_structured_fields():
    log = get_logger("deepresearch.test")
    # Should accept structlog-style kwargs whether or not structlog is installed.
    log.info("event_happened", run_id="r1", mode="fast")
    log.warning("careful", n=3)
