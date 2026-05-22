"""Embeddable agent framework public API."""
from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'core' is deprecated; use 'thund3rbot' for the public API.",
    DeprecationWarning,
    stacklevel=2,
)

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
    category=PendingDeprecationWarning,
)
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings(
        "ignore",
        message="The default value of `allowed_objects` will change",
        category=LangChainPendingDeprecationWarning,
    )
except ImportError:
    pass

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
    if name == "AgentFramework":
        from core.framework import AgentFramework

        return AgentFramework
    if name == "tool":
        from core.tooling import tool

        return tool
    if name == "prompt":
        from core.prompts import prompt

        return prompt
    raise AttributeError(f"module 'core' has no attribute {name!r}")
