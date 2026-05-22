"""FastAPI adapter for an AgentFramework runtime."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core import AgentFramework
from core.types import AgentScope, AgentSpec, ModelConfig


class RunAgentRequest(BaseModel):
    name: str
    input: str
    scope: AgentScope = AgentScope.TASK
    instructions: str = ""
    model: ModelConfig | None = None
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    max_iterations: int | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def create_agent_router(framework: AgentFramework, *, prefix: str = "/agents"):
    """Create a FastAPI router that exposes the supplied framework instance."""

    from fastapi import APIRouter, HTTPException

    router = APIRouter(prefix=prefix, tags=["agents"])

    @router.post("/run")
    async def run_agent(request: RunAgentRequest) -> dict[str, Any]:
        spec = AgentSpec(
            name=request.name,
            scope=request.scope,
            instructions=request.instructions,
            model=request.model,
            skills=request.skills,
            tools=request.tools,
            max_iterations=request.max_iterations,
            metadata=request.metadata,
        )
        result = await framework.run_agent(spec, request.input, request.context)
        return result.model_dump(mode="json")

    @router.get("/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        result = framework.runs.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        return result.model_dump(mode="json")

    @router.get("/")
    async def list_runs() -> list[dict[str, Any]]:
        return [result.model_dump(mode="json") for result in framework.runs.values()]

    return router

