"""Fixed identifiers shared across modules.

Operator-tunable values live in config.py (environment variables); these are part of the code's contracts:
names persisted in the database, strings the model or API clients see, and ADK protocol names.
"""

from enum import StrEnum

# ADK app/agent identity. Persisted sessions are keyed by the app name: changing it orphans paused runs.
APP_NAME = "ops_harness"
AGENT_NAME = "ops_assistant"

# Function call ADK emits when a tool with require_confirmation pauses for a human.
ADK_CONFIRMATION_CALL = "adk_request_confirmation"

RUN_ID_PREFIX = "run"
INCIDENT_ID_PREFIX = "INC"


class ToolName(StrEnum):
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    GET_SERVICE_STATUS = "get_service_status"
    CREATE_INCIDENT = "create_incident"


class StateKey(StrEnum):
    """Keys the harness writes into ADK session state. The `temp:` prefix is ADK's non-persisted scope."""

    TERMINATION = "harness_termination"
    CALL_COUNTS = "harness_call_counts"
    TOOL_CALLS = "harness_tool_calls"
    TOOL_ATTEMPTS_PREFIX = "temp:harness_tool_attempts:"


class TerminationReason(StrEnum):
    MAX_LLM_CALLS = "max_llm_calls_exceeded"
    MAX_TOOL_CALLS = "max_tool_calls_exceeded"
    TIME_BUDGET = "time_budget_exceeded"
    HARD_TIMEOUT = "hard_timeout"


class ToolErrorType(StrEnum):
    """`error_type` values in tool responses the harness writes for the model."""

    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    INVALID_TOOL_OUTPUT = "INVALID_TOOL_OUTPUT"
    REPEATED_CALL = "REPEATED_CALL"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    REJECTED_BY_OPERATOR = "REJECTED_BY_OPERATOR"


class TraceEventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_STATUS = "run_status"
    INVOCATION_START = "invocation_start"
    INVOCATION_END = "invocation_end"
    MODEL_CALL = "model_call"
    MODEL_ERROR = "model_error"
    TOOL_CALL = "tool_call"
    TOOL_ERROR = "tool_error"
    TOOL_INPUT_INVALID = "tool_input_invalid"
    TOOL_OUTPUT_INVALID = "tool_output_invalid"
    LOOP_DETECTED = "loop_detected"
    LIMIT_EXCEEDED = "limit_exceeded"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
