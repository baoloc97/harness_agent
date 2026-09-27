# Ops Agent Harness

An agent harness for an operations assistant, built on **Google ADK 2.10** with **Gemini (Vertex AI)** and **PostgreSQL**.
A user gives an objective ("payment-api is slow, open an incident if needed"); the agent searches runbooks, checks
service status and proposes an incident, which a human must approve before it is created.

## Design principle: use ADK where it is enough, extend it where it is not

| Requirement | Provided by ADK | Added by this harness |
|---|---|---|
| LLM ↔ tool loop | `Runner` + `LlmAgent` | – |
| State & history | `DatabaseSessionService` (Postgres) | run lifecycle table, approval audit trail |
| Tool input validation | JSON schema from the signature; arg validation runs *after* the approval gate | `ToolValidationPlugin`: validates with the same signature-derived Pydantic model **before** approval, rejects extra args |
| Tool output validation | none | `ToolValidationPlugin`: strict Pydantic output contracts |
| Tool timeouts / transient retries | none (reflect-and-retry re-prompts the model) | `resilient()` wrapper: per-attempt timeout, exponential backoff + jitter, retryable vs permanent errors |
| Tool errors reported to the model | `ReflectAndRetryToolPlugin` | – |
| Malformed LLM output | `ReflectAndRetryModelPlugin` (`MALFORMED_FUNCTION_CALL`) | run marked `failed` when retries run out |
| Transport errors (429/5xx) | google-genai `HttpRetryOptions` | – |
| Step limit | `RunConfig.max_llm_calls` | – |
| Time limit, tool budget, loop detection | none | `GuardrailPlugin` (graceful stop) + hard `asyncio.timeout` backstop |
| Approval before `create_incident` | `FunctionTool(require_confirmation=True)`, `ToolConfirmation.payload` | durable approvals, resume after restart, double-decision protection, parallel approvals, `ApprovalFeedbackPlugin` tells the model who rejected and why |
| Tracing | OpenTelemetry spans | `TracingPlugin`: per-run trace in Postgres + JSON logs |

```mermaid
flowchart LR
  CLI[CLI - Typer] --> S[HarnessService]
  API[REST API - FastAPI] --> S
  S --> R[ADK Runner]
  R --> A[LlmAgent: Gemini / RuleBased / Scripted]
  R --> P[Plugins: Tracing → ApprovalFeedback → Validation → Guardrails → ReflectRetry tool/model]
  P --> T[Tools: resilient wrapper → mock tools]
  T -. create_incident .-> G{{Approval gate}}
  S --> DB[(PostgreSQL: ADK sessions + harness tables)]
```

### Run lifecycle

```
running ──► awaiting_approval ──(approve / reject)──► running ──► completed
   │                                                       ├────► failed          (crash, model keeps returning malformed output)
   └───────────────────────────────────────────────────────┴────► limit_exceeded  (llm calls, tool calls, time budget)
```

A run is one ADK session (`session_id == run_id`). When the agent calls `create_incident`, ADK pauses the invocation;
the harness stores the approval request and returns. The approve/reject call resumes the same session, even from a
different process, because all state is in Postgres.

### Database

ADK creates its own `sessions`, `events`, `app_states`, `user_states` tables. The harness adds:

| Table | Purpose | Key columns |
|---|---|---|
| `harness_runs` | run lifecycle | id, user_id, objective, status, final_answer, error, termination_reason, llm_calls, tool_calls |
| `harness_approvals` | human-in-the-loop audit trail | id (= ADK confirmation call id), run_id, tool_name, tool_args, status, decided_by, reason, consumed |
| `harness_trace_events` | execution trace | run_id, invocation_id, ts, type, name, latency_ms, payload (JSON) |
| `mock_incidents` | stands in for the external incident system | id, title, severity, dedup_key (unique → idempotent creation) |

Trace event types: `run_started`, `invocation_start/end`, `model_call` (latency, tokens, finish reason, function calls),
`model_error`, `tool_call` (args, result, attempts, executed, latency, approved), `tool_error`, `tool_input_invalid`,
`tool_output_invalid`, `loop_detected`, `limit_exceeded`, `approval_requested`, `approval_decided`, `run_status`.

## Project layout

The package follows the layering of *Architecture Patterns with Python* (Percival & Gregory, the "Cosmic Python"
book: `domain`, `service_layer`, `adapters`, `entrypoints`, `bootstrap.py`), with the ADK-specific parts grouped under
`agent/` and tests split `unit / integration / e2e` as in Google's
[agent-starter-pack](https://github.com/GoogleCloudPlatform/agent-starter-pack).

```
harness/
  bootstrap.py        composition root: the only module that wires concrete classes together
  config.py           settings (env vars)
  constants.py        fixed identifiers: app/agent names, state keys, trace event types, error types
  views.py            read models (Pydantic), also the API response models
  observability.py    JSON logging, Tracer over a TraceSink protocol
  domain/
    model.py          RunStatus, ApprovalStatus, ids
    errors.py         NotFound / Conflict / InvalidInput (entrypoints map them to HTTP codes)
  service_layer/
    service.py        use cases: start run, decide approval, read runs / traces / incidents
    orchestrator.py   drives one ADK invocation and resolves the run's final status
  adapters/
    orm.py            SQLAlchemy tables + Database
    repository.py     RunRepository, ApprovalRepository, TraceRepository
    incidents.py      IncidentGateway protocol + MockIncidentGateway (the "external system")
    llm/              backend registry: Gemini, RuleBasedLlm (offline demo), ScriptedLlm (tests)
  agent/
    agent.py          prompt + ADK App assembly
    plugins/          tracing, approval feedback, validation, guardrails (ADK BasePlugin)
    tools/            registry, mock tools, resilience (timeout/retry), output schemas
  entrypoints/
    api.py, cli.py
data/                 mock dataset: knowledge_base.json, services.json
tests/
  unit/               pure logic, no database or model (~0.2s)
  integration/        service + database + ScriptedLlm
  e2e/                REST API and CLI
```

Dependencies point inward: `entrypoints → service_layer → domain`; adapters implement the protocols the service layer
and tools depend on (`IncidentGateway`, `TraceSink`, ADK's `BaseSessionService`).

Extension points: a new tool is a `ToolDefinition` next to its implementation; a new guardrail or observer is an ADK
plugin added in `bootstrap.py`; a real incident system implements `IncidentGateway`; a new model backend is one entry
in `LLM_BACKENDS`.

## Running

Requirements: Python 3.11+, Docker.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d --wait          # Postgres on localhost:5432
cp .env.example .env                 # then edit the LLM settings
```

**LLM backend** (`HARNESS_LLM_BACKEND`):
- `gemini`: Vertex AI. Run `gcloud auth application-default login`, then set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.
  For an AI Studio key instead, set `GOOGLE_GENAI_USE_VERTEXAI=FALSE` and `GOOGLE_API_KEY`.
- `rule_based`: deterministic offline planner, no credentials needed. Useful for demos.

### CLI

```bash
harness run "Customers report slow checkout. Investigate payment-api and open an incident if needed."
# → shows the proposed create_incident call and asks: Approve this call? [y/N]
harness show <run_id>
harness trace <run_id>
```

### API

```bash
uvicorn harness.entrypoints.api:app --reload     # OpenAPI docs at http://localhost:8000/docs
```

| Method | Path | Description |
|---|---|---|
| POST | `/runs` | start a run `{objective, user_id}`; returns when completed or paused for approval |
| GET | `/runs`, `/runs/{id}` | list runs; run detail with approvals and history |
| GET | `/runs/{id}/trace` | execution trace |
| POST | `/runs/{id}/approvals/{approval_id}` | `{decision: approve\|reject, decided_by, reason}` |
| GET | `/incidents` | incidents in the mock external system |
| POST/DELETE | `/debug/faults` | queue failures (`transient`, `timeout`, `fatal`, `malformed`) for a mock tool |

Import `postman_collection.json` into Postman; it captures `run_id` and `approval_id` automatically.

## Testing

```bash
pytest                                                            # SQLite, no network, ~3s
HARNESS_TEST_DATABASE_URL=postgresql+asyncpg://harness:harness@localhost:5432/harness_test pytest   # against Postgres (schema is reset per test)
```

Tests replace the LLM with `ScriptedLlm` so each scenario is deterministic. `pytest tests/unit` runs without a database.

| File | Covers |
|---|---|
| `unit/test_resilience.py` | retry of transient errors, no retry of permanent ones, timeouts, signature preserved, fault injector |
| `unit/test_tool_registry.py` | input model derived from the signature (constraints, enums, extra/missing args), approval flag |
| `unit/test_service_layer.py` | run status resolution, per-run locks serialize and clean up |
| `integration/test_success.py` | happy path, trace/history content, rule-based end-to-end |
| `integration/test_tool_failures.py` | transient retry, timeout exhaustion, permanent errors not retried, malformed tool output |
| `integration/test_validation.py` | invalid/extra args rejected before approval, unknown tool, malformed / empty LLM response |
| `integration/test_approval.py` | approve, reject (reason reaches the model), 409 / 404, resume after restart, parallel approvals, dedup, executed-call metric |
| `integration/test_limits.py` | max LLM calls, max tool calls, identical-call loop, wall-clock budget, hung model call (hard timeout) |
| `e2e/test_api.py` | REST flow, request validation, fault injection endpoint |
| `e2e/test_cli.py` | interactive approve / reject prompts, `show`, `trace` |

Lint and format: `ruff check harness tests && ruff format --check harness tests`.
CI (`.github/workflows/ci.yml`) runs lint plus the test suite on SQLite and on a Postgres service container.

## Environment variables

See `.env.example`. Harness settings use the `HARNESS_` prefix and are validated at startup (`harness/config.py`):
an out-of-range value stops the app with a clear error instead of misbehaving at run time.

| Variable | Default | Meaning |
|---|---|---|
| `HARNESS_DATABASE_URL` | `postgresql+asyncpg://harness:harness@localhost:5432/harness` | shared by ADK sessions and harness tables |
| **LLM** | | |
| `HARNESS_LLM_BACKEND` | `gemini` | `gemini` or `rule_based` |
| `HARNESS_MODEL` | `gemini-2.5-flash` | Gemini model id |
| `HARNESS_MODEL_TEMPERATURE` | 0.2 | low for consistent operational behaviour (0–2) |
| `HARNESS_MODEL_HTTP_RETRY_ATTEMPTS` / `_INITIAL_DELAY_S` | 3 / 1.0 | transport retries for 429/5xx |
| `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_API_KEY` | – | Google credentials (read by ADK) |
| **Execution limits** | | |
| `HARNESS_MAX_LLM_CALLS` | 12 | ADK `RunConfig.max_llm_calls` |
| `HARNESS_MAX_TOOL_CALLS` | 10 | tool call budget per run |
| `HARNESS_MAX_IDENTICAL_TOOL_CALLS` | 2 | further identical calls are blocked |
| `HARNESS_RUN_TIME_BUDGET_S` | 60 | wall-clock budget per invocation (approval wait excluded) |
| `HARNESS_MAX_MODEL_RETRIES` | 2 | malformed model responses tolerated |
| **Tool resilience** | | |
| `HARNESS_TOOL_TIMEOUT_S` / `_MAX_ATTEMPTS` / `_BACKOFF_BASE_S` | 5 / 3 / 0.5 | per-attempt timeout, attempts, exponential backoff base |
| `HARNESS_TOOL_REFLECTION_RETRIES` | 1 | times the model is asked to reflect and retry after a tool error |
| **Mock tools** | | |
| `HARNESS_DATA_DIR` | `./data` | location of the mock dataset |
| `HARNESS_KB_TOP_K` | 3 | knowledge base results returned per search |
| `HARNESS_MOCK_FAULTS` | – | e.g. `get_service_status=transient,timeout` |
| **Observability** | | |
| `HARNESS_LOG_LEVEL` | INFO | `DEBUG` / `INFO` / `WARNING` / `ERROR`, JSON logs on stderr |
| `HARNESS_TRACE_MAX_STR_LEN` | 2000 | truncation of long strings in trace payloads |

Deliberately kept in code: the ADK app and agent names (they key persisted sessions, so changing them orphans paused
runs), the system prompt (reviewed and versioned with the code), and tool argument constraints (the contract the
model is given). `GET /runs` takes `?limit=` (1–500) per request instead of a global setting.

## Limitations and future work

- **Synchronous API**: `POST /runs` holds the request until the run pauses or finishes. Next step: run in a worker
  (Cloud Tasks / Celery / Pub/Sub), return `202` and stream progress over SSE.
- **Single process**: the per-run lock is in memory; approvals themselves are safe (atomic DB update), but two API
  replicas could resume the same run concurrently. Use a Postgres advisory lock or Redis lock.
- **No auth / RBAC**: anyone can approve. Add identity (IAP / OAuth), approver roles and severity-based policies
  (e.g. `critical` needs two approvers), plus approval expiry.
- **Trace writes are synchronous** per event. Batch them, or export ADK's OpenTelemetry spans to Cloud Trace.
- **Prompt injection**: runbook text reaches the model unfiltered. Add content sanitization and treat tool output as data.
- **Schema management** uses `create_all` at startup; production needs Alembic migrations.
- **Time budget is per invocation**, so time spent waiting for a human is excluded, but a run with many approvals can
  use several budgets. A per-run cumulative active-time budget would be stricter.
- **Mock search** is keyword scoring. A real system would use Vertex AI Search or pgvector embeddings.
- **Evaluation**: add `adk eval` sets to measure answer quality and tool-choice accuracy against the real model, and
  cost/token budgets alongside the call budgets.
