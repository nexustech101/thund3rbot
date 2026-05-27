"""Embeddable agent framework runtime."""
from __future__ import annotations

import inspect
import time
from typing import Any

from thund3rbot.agents import OrchestratorAgent, SubAgent, TaskAgent
from thund3rbot.memory import BaseMemoryStore, InMemoryStore
from thund3rbot.models import create_llm
from thund3rbot.prompts import PromptRegistry
from thund3rbot.skills import SkillRegistry
from thund3rbot.tooling import ToolRegistry
from thund3rbot.types import (
    AgentInput,
    AgentScope,
    AgentSpec,
    FactoryConfig,
    FactoryEvent,
    ModelConfig,
    RunOptions,
    RunResult,
    WorkflowStepFinished,
    WorkflowStepStarted,
)
from thund3rbot.workflows import WorkflowRegistry


class AgentFactory:
    """A self-contained runtime for creating agents, tools, skills, and workflows."""

    def __init__(
        self,
        config: FactoryConfig,
        *,
        memory: BaseMemoryStore | None = None,
    ) -> None:
        self.config = config
        self.memory = memory or InMemoryStore()
        self.tools = ToolRegistry(self)
        self.prompts = PromptRegistry(self)
        self.skills = SkillRegistry(self)
        self.workflows = WorkflowRegistry(self)
        self.runs: dict[str, RunResult] = {}
        self.scope_prompts: dict[AgentScope, str] = {}
        self._event_hooks: list[Any] = []
        self._active_workflow: str = ""

        if self.config.enable_default_tools:
            self._register_default_tools()

    def agent(self, spec: AgentSpec) -> TaskAgent | SubAgent | OrchestratorAgent:
        """Create an agent handle from a developer-facing spec."""

        if spec.scope == AgentScope.ORCHESTRATOR:
            return OrchestratorAgent(self, spec)
        if spec.scope == AgentScope.SUB_AGENT:
            return SubAgent(self, spec)
        return TaskAgent(self, spec)

    async def run_agent(
        self,
        spec: AgentSpec,
        input: AgentInput,
        context: dict[str, Any] | None = None,
        *,
        options: RunOptions | None = None,
    ) -> RunResult:
        return await self.agent(spec).run(input, context=context, options=options)

    async def step(
        self,
        name: str,
        spec: AgentSpec,
        input: AgentInput,
        *,
        context: dict[str, Any] | None = None,
        options: RunOptions | None = None,
    ) -> RunResult:
        started = time.perf_counter()
        await self.emit(
            WorkflowStepStarted(
                run_id=spec.agent_id,
                agent_name=spec.name,
                workflow=self._active_workflow,
                step=name,
            )
        )
        try:
            return await self.run_agent(spec, input, context=context, options=options)
        finally:
            await self.emit(
                WorkflowStepFinished(
                    run_id=spec.agent_id,
                    agent_name=spec.name,
                    workflow=self._active_workflow,
                    step=name,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            )

    def create_model(self, config: ModelConfig | None = None) -> Any:
        model_config = config or self.config.default_model
        if self.config.model_factory:
            return self.config.model_factory(model_config)
        return create_llm(model_config, self.config.providers)

    def workflow(self, name: str, handler=None, *, description: str = ""):
        """Shortcut for ``framework.workflows.register``."""

        return self.workflows.register(name, handler, description=description)

    def on_event(self, hook):
        """Register a sync or async framework event hook."""

        self._event_hooks.append(hook)
        return hook

    async def emit(self, event: FactoryEvent) -> None:
        for hook in list(self._event_hooks):
            value = hook(event)
            if inspect.isawaitable(value):
                await value

    def _register_default_tools(self) -> None:
        @self.tools.register(scopes=[AgentScope.TASK, AgentScope.SUB_AGENT, AgentScope.ORCHESTRATOR])
        def echo(text: str) -> str:
            """Return text unchanged."""

            return text

        @self.tools.register(scopes=[AgentScope.TASK, AgentScope.SUB_AGENT, AgentScope.ORCHESTRATOR])
        def merge_texts(texts: list[str], separator: str = "\n") -> str:
            """Merge text fragments with a separator."""

            return separator.join(texts)
