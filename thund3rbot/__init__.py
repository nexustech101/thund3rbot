"""Thund3rBot public API."""
from __future__ import annotations

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from core.types import (
        AgentInput,
        AgentScope,
        AgentSpec,
        AgentStatus,
        AgentFinished,
        AgentStarted,
        Artifact,
        ContentPart,
        FrameworkConfig,
        FrameworkConfigError,
        FrameworkEvent,
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
    "AgentFramework",
    "AgentInput",
    "AgentScope",
    "AgentSpec",
    "AgentStatus",
    "AgentFinished",
    "AgentStarted",
    "Artifact",
    "ContentPart",
    "FrameworkConfig",
    "FrameworkConfigError",
    "FrameworkEvent",
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
        if name == "AgentFramework":
            from core.framework import AgentFramework

            return AgentFramework
        if name == "tool":
            from core.tooling import tool

            return tool
        if name == "prompt":
            from core.prompts import prompt

            return prompt
    raise AttributeError(f"module 'thund3rbot' has no attribute {name!r}")
