"""Structured JSON logging + a tracer that persists every harness event to the database."""

from __future__ import annotations

import logging
import sys
import warnings
from typing import Any, Protocol

import structlog

log = structlog.get_logger("harness.trace")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(stream=sys.stderr, level=level, format="%(message)s")
    # ADK and google-genai are chatty; their events are already captured by the tracer.
    for noisy in ("google_adk", "google_genai", "httpx", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    # ADK flags tool confirmation etc. as experimental on every use.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.adk.*")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


class TraceSink(Protocol):
    """Where trace events are persisted (Postgres today; could be an OpenTelemetry exporter)."""

    async def add(
        self,
        run_id: str,
        type: str,
        name: str | None,
        invocation_id: str | None,
        latency_ms: float | None,
        payload: dict[str, Any],
    ) -> None: ...


class Tracer:
    """Emits one trace event: a JSON log line plus a record in the sink."""

    def __init__(self, sink: TraceSink, *, max_str_len: int = 2000):
        self._sink = sink
        self._max_str_len = max_str_len

    async def emit(
        self,
        run_id: str,
        type: str,
        *,
        name: str | None = None,
        invocation_id: str | None = None,
        latency_ms: float | None = None,
        **payload: Any,
    ) -> None:
        payload = _jsonable(payload, self._max_str_len)
        log.info(type, run_id=run_id, name=name, invocation_id=invocation_id, latency_ms=latency_ms, **payload)
        try:
            await self._sink.add(run_id, type, name, invocation_id, latency_ms, payload)
        except Exception as e:  # tracing must never break a run
            log.error("trace_persist_failed", run_id=run_id, error=str(e))


def _jsonable(value: Any, max_str: int) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v, max_str) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, max_str) for v in value]
    if isinstance(value, str):
        return value if len(value) <= max_str else value[:max_str] + "…"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _jsonable(str(value), max_str)
