"""
Agent and workflow lifecycle registry.

Both registries are process-global singletons.  They track status and results
for every agent and workflow spawned during a process lifetime.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from thund3rbot.types import AgentResult, AgentScope, AgentStatus, WorkflowResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Registry
# ─────────────────────────────────────────────────────────────────────────────


class AgentRegistry:
    """Thread-safe store of AgentResult objects keyed by agent_id."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentResult] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def register(self, agent_id: str, name: str, scope: AgentScope) -> AgentResult:
        result = AgentResult(agent_id=agent_id, name=name, scope=scope)
        self._agents[agent_id] = result
        self._locks[agent_id] = asyncio.Lock()
        logger.debug("Registered agent %s (%s)", name, agent_id)
        return result

    async def update(
        self,
        agent_id: str,
        status: AgentStatus,
        output: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        lock = self._locks.get(agent_id) or asyncio.Lock()
        async with lock:
            result = self._agents.get(agent_id)
            if result is None:
                return
            result.status = status
            if output is not None:
                result.output = output
            if error is not None:
                result.error = error
            if metadata:
                result.metadata.update(metadata)
            if status in {AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED}:
                result.completed_at = datetime.utcnow()

    def get(self, agent_id: str) -> Optional[AgentResult]:
        return self._agents.get(agent_id)

    def list_all(self) -> list[AgentResult]:
        return list(self._agents.values())

    def list_by_status(self, status: AgentStatus) -> list[AgentResult]:
        return [a for a in self._agents.values() if a.status == status]


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Registry
# ─────────────────────────────────────────────────────────────────────────────


class WorkflowRegistry:
    """Store of WorkflowResult objects keyed by workflow_id."""

    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowResult] = {}

    def register(self, workflow_id: str, workflow_name: str = "") -> WorkflowResult:
        result = WorkflowResult(workflow_id=workflow_id, workflow_name=workflow_name)
        self._workflows[workflow_id] = result
        logger.debug("Registered workflow %s (%s)", workflow_name, workflow_id)
        return result

    def get(self, workflow_id: str) -> Optional[WorkflowResult]:
        return self._workflows.get(workflow_id)

    def list_all(self) -> list[WorkflowResult]:
        return list(self._workflows.values())

    async def update(self, workflow_id: str, **kwargs: Any) -> None:
        result = self._workflows.get(workflow_id)
        if result is None:
            return
        for key, value in kwargs.items():
            setattr(result, key, value)
        if kwargs.get("status") in {
            AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED
        }:
            result.completed_at = datetime.utcnow()


# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

agent_registry = AgentRegistry()
workflow_registry = WorkflowRegistry()