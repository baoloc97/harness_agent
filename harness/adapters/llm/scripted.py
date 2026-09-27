"""ScriptedLlm: replays a fixed list of responses; used by the tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from pydantic import PrivateAttr

from harness.adapters.llm.responses import text

Step = LlmResponse | Callable[[LlmRequest], LlmResponse]


class ScriptedLlm(BaseLlm):
    model: str = "scripted"
    _steps: list[Step] = PrivateAttr(default_factory=list)
    _requests: list[LlmRequest] = PrivateAttr(default_factory=list)
    _delay_s: float = PrivateAttr(default=0.0)

    def __init__(self, steps: list[Step], delay_s: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self._steps = list(steps)
        self._delay_s = delay_s

    @property
    def requests(self) -> list[LlmRequest]:
        return self._requests

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self._requests.append(llm_request)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if not self._steps:
            yield text("(script exhausted)")
            return
        step = self._steps.pop(0)
        yield step(llm_request) if callable(step) else step
