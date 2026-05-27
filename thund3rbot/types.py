"""Public framework types.

The new API is intentionally centered on embeddable application use.  The
legacy CLI/API models are kept as thin aliases where practical so older sample
apps can be adapted gradually.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Callable, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


class AgentScope(str, Enum):
    """Controls the framework capabilities granted to an agent."""
    ORCHESTRATOR = "orchestrator"
    SUB_AGENT = "sub_agent"
    TASK = "task_agent"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProviderConfig(BaseModel):
    """Provider-level defaults and credentials."""

    name: str
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuration for a single chat model.

    ``model`` is the public field.  ``model_name`` is accepted for compatibility
    with the previous codebase and common LangChain examples.
    """

    model_config = ConfigDict(populate_by_name=True)

    provider: ModelProvider | str = ModelProvider.OLLAMA
    model: str = Field(
        "llama3.2",
        validation_alias=AliasChoices("model", "model_name"),
        serialization_alias="model",
    )
    temperature: float = 0.7
    max_tokens: int | None = None
    streaming: bool = False
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def model_name(self) -> str:
        """Compatibility property for older callers."""

        return self.model


ModelFactory = Callable[[ModelConfig], Any]
WorkflowHandler = Callable[[dict[str, Any]], Any] | Callable[[dict[str, Any], Any], Any]
ToolRef = str | Callable[..., Any] | Any
StopReason = Literal["completed", "max_steps", "max_tool_calls", "timeout", "error"]
ToolRisk = Literal["low", "medium", "high"]
ContentType = Literal["text", "image", "audio", "video", "html", "markdown", "json", "file", "screenshot"]


class ContentPart(BaseModel):
    """Typed input content for multimodal-capable agents."""

    type: ContentType = "text"
    text: str | None = None
    data: Any = None
    uri: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    """Typed output or intermediate asset produced by an agent or tool."""

    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ContentType
    data: Any = None
    uri: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


AgentInput = str | list[ContentPart]


class FactoryConfig(BaseModel):
    """Top-level configuration for an ``AgentFactory`` runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_model: ModelConfig = Field(default_factory=ModelConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    model_factory: ModelFactory | None = None
    max_iterations: int = Field(10, ge=1)
    enable_default_tools: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Skill(BaseModel):
    """Prompt/tool bundle that can be attached to an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    instructions: str = ""
    tools: list[ToolRef] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    scope: AgentScope | None = None
    scopes: set[AgentScope] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    """Metadata for a registered tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    scopes: set[AgentScope] = Field(default_factory=set)
    tool: Any
    source: Callable[..., Any] | None = None
    public_name: str | None = None
    namespace: str | None = None
    risk: ToolRisk = "low"
    requires_approval: bool = False
    tags: set[str] = Field(default_factory=set)


class PromptSpec(BaseModel):
    """Prompt template registered with the factory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    prompt: Callable[..., str] | str
    namespace: str | None = None


class AgentSpec(BaseModel):
    """Developer-facing agent declaration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    name: str
    scope: AgentScope = AgentScope.TASK
    instructions: str = ""
    model: ModelConfig | None = None
    skills: list[str] = Field(default_factory=list)
    tools: list[ToolRef] = Field(default_factory=list)
    output_schema: type[BaseModel] | None = None
    max_iterations: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowSpec(BaseModel):
    """Metadata for a registered workflow/pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    handler: WorkflowHandler


class RunResult(BaseModel):
    """Structured result returned by agents and workflows."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    session_id: str = ""
    name: str = ""
    scope: AgentScope | None = None
    status: AgentStatus = AgentStatus.PENDING
    output: Any = None
    error: str | None = None
    stop_reason: StopReason | None = None
    steps: int = 0
    tool_calls: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    child_results: list["RunResult"] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class StepEvent(BaseModel):
    """Per-step callback payload."""

    index: int
    tool: str | None = None
    input: Any = None
    output: Any = None
    summary: str = ""


class ToolCallContext(BaseModel):
    """Payload passed to tool lifecycle hooks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    agent_id: str
    session_id: str
    agent_name: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk: ToolRisk = "low"
    requires_approval: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolApproval(BaseModel):
    """Decision returned by ``before_tool_call`` hooks."""

    approved: bool = True
    arguments: dict[str, Any] | None = None
    reason: str | None = None


class RunOptions(BaseModel):
    """Per-run controls separate from static ``AgentSpec`` configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_steps: int | None = None
    max_tool_calls: int | None = None
    timeout_seconds: float | None = None
    on_step: Callable[[StepEvent], Any] | None = None
    before_tool_call: Callable[[ToolCallContext], Any] | None = None
    after_tool_call: Callable[[ToolCallContext, Any], Any] | None = None
    on_tool_error: Callable[[ToolCallContext, Exception], Any] | None = None


class FactoryEvent(BaseModel):
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    agent_name: str


class AgentStarted(FactoryEvent):
    event_type: str = "agent_started"
    scope: AgentScope
    input: Any


class ToolCalled(FactoryEvent):
    event_type: str = "tool_called"
    tool: str
    input: Any = None


class ToolResult(FactoryEvent):
    event_type: str = "tool_result"
    tool: str
    output: Any = None
    duration_ms: float = 0.0


class AgentFinished(FactoryEvent):
    event_type: str = "agent_finished"
    output: Any = None
    stop_reason: StopReason | None = None
    steps: int = 0
    duration_ms: float = 0.0


class WorkflowStepStarted(FactoryEvent):
    event_type: str = "workflow_step_started"
    workflow: str = ""
    step: str


class WorkflowStepFinished(FactoryEvent):
    event_type: str = "workflow_step_finished"
    workflow: str = ""
    step: str
    duration_ms: float = 0.0


class FactoryConfigError(ValueError):
    """Base factory configuration error."""


class ToolNotFoundError(FactoryConfigError):
    """Raised when a named tool cannot be resolved."""


class SkillConfigError(FactoryConfigError):
    """Raised when skill registration or composition is invalid."""


# Compatibility models retained for older sample app modules.
class AgentConfig(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    scope: AgentScope
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    system_prompt: str = ""
    tool_names: list[str] = Field(default_factory=list)
    max_iterations: int = 10
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    agent_id: str
    name: str = ""
    scope: Optional[AgentScope] = None
    status: AgentStatus = AgentStatus.PENDING
    output: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: Optional[datetime] = None


class WorkflowConfig(BaseModel):
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    workflow_id: str
    workflow_name: str = ""
    status: AgentStatus = AgentStatus.PENDING
    output: Optional[Any] = None
    error: Optional[str] = None
    agent_results: list[AgentResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: Optional[datetime] = None
