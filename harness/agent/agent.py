"""The ops assistant agent and its ADK App."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.plugins import BasePlugin
from google.adk.tools import BaseTool
from google.genai import types

from harness.constants import AGENT_NAME, APP_NAME

INSTRUCTION = """You are an operations assistant for an SRE team.
Work only from tool results; never invent service data, runbooks or incident ids.
Procedure:
1. Search the knowledge base for relevant runbooks and the severity guidelines.
2. Check the status of every service the objective concerns.
3. Create an incident only if a service is degraded or down, choosing severity from the guidelines.
   A human must approve incident creation; if it is rejected, do not retry it.
4. If a tool returns an error, do not repeat the identical call; adapt or explain the gap.
Finish with a concise answer: findings, evidence (status numbers, runbook ids) and actions taken."""


def build_app(llm: BaseLlm, tools: list[BaseTool], plugins: list[BasePlugin], *, temperature: float) -> App:
    agent = LlmAgent(
        name=AGENT_NAME,
        model=llm,
        instruction=INSTRUCTION,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(temperature=temperature),
    )
    return App(name=APP_NAME, root_agent=agent, plugins=plugins)
