"""HTTP API. Run with: uvicorn harness.entrypoints.api:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from harness.bootstrap import Harness, build_harness
from harness.config import get_settings
from harness.constants import ToolName
from harness.domain.errors import ConflictError, HarnessError, InvalidInputError, NotFoundError
from harness.observability import configure_logging
from harness.service_layer.service import HarnessService
from harness.views import IncidentView, RunSummary, RunView, TraceEventView

HTTP_STATUS: dict[type[HarnessError], int] = {InvalidInputError: 400, NotFoundError: 404, ConflictError: 409}


class CreateRunRequest(BaseModel):
    objective: str = Field(
        min_length=1, max_length=2000, examples=["payment-api is slow, investigate and open an incident if needed"]
    )
    user_id: str = Field(default="default", max_length=100)


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]
    decided_by: str = Field(default="operator", max_length=100)
    reason: str | None = Field(default=None, max_length=1000)


class FaultRequest(BaseModel):
    tool: ToolName
    faults: list[Literal["transient", "timeout", "fatal", "malformed"]]


def create_app(harness: Harness | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        configure_logging(settings.log_level)
        h = harness or build_harness(settings)
        await h.start()
        app.state.harness = h
        yield
        await h.stop()

    app = FastAPI(title="Ops Agent Harness", version="0.2.0", lifespan=lifespan)

    @app.exception_handler(HarnessError)
    async def _harness_error(_: Request, exc: HarnessError) -> JSONResponse:
        return JSONResponse(status_code=HTTP_STATUS.get(type(exc), 400), content={"detail": str(exc)})

    def svc(request: Request) -> HarnessService:
        return request.app.state.harness.service

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/runs", status_code=201, response_model=RunView)
    async def create_run(body: CreateRunRequest, request: Request) -> RunView:
        """Starts a run and executes it until it completes, fails, hits a limit, or needs approval."""
        return await svc(request).start_run(body.objective, body.user_id)

    @app.get("/runs", response_model=list[RunSummary])
    async def list_runs(request: Request, limit: int = Query(50, ge=1, le=500)) -> list[RunSummary]:
        return await svc(request).list_runs(limit)

    @app.get("/runs/{run_id}", response_model=RunView)
    async def get_run(run_id: str, request: Request) -> RunView:
        return await svc(request).get_run(run_id)

    @app.get("/runs/{run_id}/trace", response_model=list[TraceEventView])
    async def get_trace(run_id: str, request: Request) -> list[TraceEventView]:
        return await svc(request).get_trace(run_id)

    @app.post("/runs/{run_id}/approvals/{approval_id}", response_model=RunView)
    async def decide(run_id: str, approval_id: str, body: ApprovalDecision, request: Request) -> RunView:
        """Approves or rejects a pending tool call; the run resumes once every pending approval is decided."""
        return await svc(request).decide_approval(
            run_id, approval_id, body.decision == "approve", body.decided_by, body.reason
        )

    @app.get("/incidents", response_model=list[IncidentView])
    async def incidents(request: Request) -> list[IncidentView]:
        return await svc(request).list_incidents()

    @app.post("/debug/faults", status_code=202)
    async def inject_faults(body: FaultRequest, request: Request) -> dict:
        """Demo only: queue failures for the next calls of a mock tool."""
        request.app.state.harness.faults.add(body.tool, *body.faults)
        return {"queued": body.model_dump()}

    @app.delete("/debug/faults", status_code=204)
    async def clear_faults(request: Request) -> None:
        request.app.state.harness.faults.clear()

    return app


app = create_app()
