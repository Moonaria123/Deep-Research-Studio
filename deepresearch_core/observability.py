"""Structured logging + optional OpenTelemetry for the baseline.

Uses ``structlog`` when installed for JSON structured logs, otherwise falls back to
the standard library ``logging`` module. OpenTelemetry tracing is wired only when the
SDK and an exporter endpoint are available; otherwise tracing is a no-op. Secrets are
never logged -- callers must pass already-redacted values.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

_CONFIGURED = False


def configure_logging(*, level: str | None = None, service_name: str = "deepresearch") -> None:
    """Configure structured logging once for the process.

    Honors ``DEEPRESEARCH_LOG_LEVEL`` (default ``INFO``). When ``structlog`` is present,
    logs are emitted as JSON; otherwise a plain stdlib formatter is used.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = (level or os.environ.get("DEEPRESEARCH_LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, resolved, logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    try:
        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            cache_logger_on_first_use=True,
        )
    except Exception:
        pass

    _maybe_setup_otel(service_name)
    _CONFIGURED = True


class _StdlibLogger:
    """Stdlib-backed logger that accepts structlog-style keyword fields.

    Keyword fields are rendered as ``key=value`` pairs appended to the event message,
    so call sites can use the structured API uniformly whether or not structlog is present.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        if fields:
            extras = " ".join(f"{k}={v}" for k, v in fields.items())
            self._logger.log(level, "%s %s", event, extras)
        else:
            self._logger.log(level, "%s", event)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._logger.exception("%s %s", event, " ".join(f"{k}={v}" for k, v in fields.items()))


def get_logger(name: str = "deepresearch") -> Any:
    """Return a structlog logger when available, else a kwargs-tolerant stdlib logger."""

    try:
        import structlog

        return structlog.get_logger(name)
    except Exception:
        return _StdlibLogger(logging.getLogger(name))


def _maybe_setup_otel(service_name: str) -> None:
    """Wire OpenTelemetry tracing only if the SDK + an OTLP endpoint are present."""

    endpoint = str(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
    except Exception:
        # Tracing is best-effort; never block the app on telemetry wiring.
        pass


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Start an OpenTelemetry span when tracing is active; otherwise a no-op.

    Only the tracer *setup* is guarded by the fallback; exceptions raised by the wrapped
    body always propagate unchanged (the span records and re-raises them). Guarding the body
    here would corrupt control flow -- e.g. swallowing a cooperative ``RUN_CANCELLED`` and
    double-yielding from this generator.
    """

    tracer = None
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("deepresearch")
    except Exception:
        tracer = None

    if tracer is None:
        yield
        return

    with tracer.start_as_current_span(name) as current:
        try:
            for key, value in attributes.items():
                current.set_attribute(key, value)
        except Exception:
            pass
        yield
