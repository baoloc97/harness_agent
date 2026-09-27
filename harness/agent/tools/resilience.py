"""Timeout + retry wrapper for tools, and a fault injector for the mocks.

ADK has no per-tool timeout and no transparent retry of transient failures
(ReflectAndRetryToolPlugin retries by asking the *model* to call again). This
wrapper handles transient failures inside a single tool call, so the model only
sees errors that retrying cannot fix.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from harness.constants import StateKey

log = structlog.get_logger("harness.tools")


def attempts_key(function_call_id: str | None) -> str:
    """Per-call key: parallel tool calls in one model turn must not overwrite each other."""
    return f"{StateKey.TOOL_ATTEMPTS_PREFIX}{function_call_id}"


class ToolError(Exception):
    retryable = False


class TransientToolError(ToolError):
    """Upstream hiccup (e.g. HTTP 503); safe to retry."""

    retryable = True


class ToolTimeoutError(ToolError):
    retryable = True


class PermanentToolError(ToolError):
    """Retrying will not help (bad request, unknown entity, bug)."""


class ToolRetriesExhausted(ToolError):
    def __init__(self, tool: str, attempts: int, last_error: Exception):
        super().__init__(f"{tool} failed after {attempts} attempts: {type(last_error).__name__}: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


@dataclass(frozen=True)
class RetryPolicy:
    timeout_s: float
    max_attempts: int
    backoff_base_s: float


def resilient(func: Callable[..., Awaitable[Any]], policy: RetryPolicy) -> Callable[..., Awaitable[Any]]:
    """Wraps an async tool: per-attempt timeout, exponential backoff with jitter on retryable errors.

    functools.wraps keeps the original signature, which ADK uses to build the tool declaration.
    """
    timeout_s, max_attempts, backoff_base_s = policy.timeout_s, policy.max_attempts, policy.backoff_base_s

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_context = kwargs.get("tool_context")
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_s)
                _record_attempts(tool_context, attempt)
                return result
            except TimeoutError:
                last_error = ToolTimeoutError(f"timed out after {timeout_s}s")
            except ToolError as e:
                last_error = e
            log.warning(
                "tool_attempt_failed",
                tool=func.__name__,
                attempt=attempt,
                error=str(last_error),
                retryable=last_error.retryable,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            if not last_error.retryable:
                _record_attempts(tool_context, attempt)
                raise last_error
            if attempt < max_attempts:
                delay = backoff_base_s * (2 ** (attempt - 1))
                await asyncio.sleep(delay + random.uniform(0, delay / 2))
        _record_attempts(tool_context, max_attempts)
        raise ToolRetriesExhausted(func.__name__, max_attempts, last_error)

    return wrapper


def _record_attempts(tool_context: Any, attempts: int) -> None:
    if tool_context is not None:
        tool_context.state[attempts_key(tool_context.function_call_id)] = attempts


class FaultInjector:
    """Queues faults per tool; each tool attempt consumes one.

    Kinds: transient, timeout, fatal, malformed.
    Config string format: "tool_a=transient,timeout;tool_b=fatal".
    """

    KINDS = {"transient", "timeout", "fatal", "malformed"}

    def __init__(self) -> None:
        self._queues: dict[str, deque[str]] = defaultdict(deque)

    @classmethod
    def from_string(cls, spec: str) -> FaultInjector:
        injector = cls()
        for chunk in filter(None, (c.strip() for c in spec.split(";"))):
            tool, _, kinds = chunk.partition("=")
            injector.add(tool.strip(), *(k.strip() for k in kinds.split(",") if k.strip()))
        return injector

    def add(self, tool: str, *kinds: str) -> None:
        for kind in kinds:
            if kind not in self.KINDS:
                raise ValueError(f"unknown fault kind {kind!r}")
            self._queues[tool].append(kind)

    def clear(self) -> None:
        self._queues.clear()

    async def apply(self, tool: str) -> str | None:
        """Raises/sleeps for the next fault; returns 'malformed' so the tool can corrupt its output."""
        if not self._queues[tool]:
            return None
        kind = self._queues[tool].popleft()
        if kind == "transient":
            raise TransientToolError("injected: upstream returned 503")
        if kind == "fatal":
            raise PermanentToolError("injected: upstream returned 400")
        if kind == "timeout":
            await asyncio.sleep(3600)
        return kind
