"""Thund3rBot public API."""
from __future__ import annotations

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from thund3rbot.types import (
        AgentInput,
        AgentScope,
        AgentSpec,
        AgentStatus,
        AgentFinished,
        AgentStarted,
        Artifact,
        ContentPart,
        FactoryConfig,
        FactoryConfigError,
        FactoryEvent,
        ModelConfig,
        ModelProvider,
        PromptSpec,
        ProviderConfig,
        RunOptions,
        RunResult,
        Skill,
        SkillConfigError,
        StepEvent,
        ToolApproval,
        ToolCalled,
        ToolCallContext,
        ToolNotFoundError,
        ToolRisk,
        ToolResult,
        ToolSpec,
        WorkflowStepFinished,
        WorkflowStepStarted,
        WorkflowSpec,
    )

__all__ = [
    "AgentFactory",
    "AgentInput",
    "AgentScope",
    "AgentSpec",
    "AgentStatus",
    "AgentFinished",
    "AgentStarted",
    "Artifact",
    "ContentPart",
    "FactoryConfig",
    "FactoryConfigError",
    "FactoryEvent",
    "ModelConfig",
    "ModelProvider",
    "PromptSpec",
    "ProviderConfig",
    "RunOptions",
    "RunResult",
    "Skill",
    "SkillConfigError",
    "StepEvent",
    "ToolApproval",
    "ToolCalled",
    "ToolCallContext",
    "ToolNotFoundError",
    "ToolRisk",
    "ToolResult",
    "ToolSpec",
    "WorkflowStepFinished",
    "WorkflowStepStarted",
    "WorkflowSpec",
    "prompt",
    "tool",
]


def __getattr__(name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if name == "AgentFactory":
            from thund3rbot.factory import AgentFactory

            return AgentFactory
        if name == "tool":
            from thund3rbot.tooling import tool

            return tool
        if name == "prompt":
            from thund3rbot.prompts import prompt

            return prompt
    raise AttributeError(f"module 'thund3rbot' has no attribute {name!r}")
