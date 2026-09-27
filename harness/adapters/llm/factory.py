"""LLM backend registry. Callers only ever use build_llm()."""

from __future__ import annotations

from collections.abc import Callable

from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.genai import types

from harness.adapters.llm.rule_based import RuleBasedLlm
from harness.config import Settings


def _gemini(settings: Settings) -> BaseLlm:
    # Transport-level retries (429/5xx) are built into google-genai; configure rather than reimplement.
    retry = types.HttpRetryOptions(
        attempts=settings.model_http_retry_attempts, initial_delay=settings.model_http_retry_initial_delay_s
    )
    return Gemini(model=settings.model, retry_options=retry)


# Register new backends here.
LLM_BACKENDS: dict[str, Callable[[Settings], BaseLlm]] = {
    "gemini": _gemini,
    "rule_based": lambda _: RuleBasedLlm(),
}


def build_llm(settings: Settings) -> BaseLlm:
    return LLM_BACKENDS[settings.llm_backend](settings)
