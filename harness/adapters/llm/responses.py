"""Helpers to build model responses (used by the offline backends and the tests)."""

from __future__ import annotations

from typing import Any

from google.adk.models.llm_response import LlmResponse
from google.genai import types


def call(name: str, **args: Any) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])
    )


def text(value: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=value)]))


def malformed() -> LlmResponse:
    """What Gemini returns when it emits an unparsable function call."""
    return LlmResponse(
        error_code="MALFORMED_FUNCTION_CALL",
        error_message="Malformed function call: create_incident(title=",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )
