"""In-process workflow/pipeline registry."""
from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any

from core.types import AgentStatus, RunResult, WorkflowHandler, WorkflowSpec


class WorkflowRegistry:
    """Runtime-local workflow registry."""

    def __init__(self, framework: Any) -> None:
        self._framework = framework
        self._workflows: dict[str, WorkflowSpec] = {}

    def register(
        self,
        name: str,
        handler: WorkflowHandler | None = None,
        *,
        description: str = "",
    ):
        """Register a workflow handler directly or as a decorator."""

        def _register(fn: WorkflowHandler):
            self._workflows[name] = WorkflowSpec(name=name, description=description, handler=fn)
            return fn

        if handler is None:
            return _register
        return _register(handler)

    def get(self, name: str) -> WorkflowSpec | None:
        return self._workflows.get(name)

    def list(self) -> list[WorkflowSpec]:
        return list(self._workflows.values())

    async def run(self, name: str, context: dict[str, Any] | None = None) -> RunResult:
        workflow = self.get(name)
        if workflow is None:
            available = ", ".join(sorted(self._workflows)) or "(none)"
            raise KeyError(f"Workflow {name!r} is not registered. Available: {available}")

        result = RunResult(name=name, status=AgentStatus.RUNNING, metadata={"kind": "workflow"})
        previous_workflow = self._framework._active_workflow
        self._framework._active_workflow = name
        try:
            output = await self._invoke(workflow.handler, context or {})
            result.output = output
            result.status = AgentStatus.COMPLETED
        except Exception as exc:
            result.error = str(exc)
            result.status = AgentStatus.FAILED
        finally:
            self._framework._active_workflow = previous_workflow
            result.completed_at = datetime.now(UTC)
            self._framework.runs[result.run_id] = result
        return result

    async def _invoke(self, handler: WorkflowHandler, context: dict[str, Any]) -> Any:
        params = inspect.signature(handler).parameters
        output = handler(context, self._framework) if len(params) >= 2 else handler(context)  # type: ignore[misc]
        if inspect.isawaitable(output):
            return await output
        return output
