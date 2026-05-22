# Thund3rBot

Thund3rBot is a tiny embeddable Python framework for building scoped agents inside
applications, APIs, scripts, CLIs, and automation services. It gives developers a
small runtime kernel: model configuration, instructions, tools, skills, context,
memory, workflows, run tracking, and lifecycle hooks.

The core package is intentionally lean. Provider SDKs, FastAPI, MCP/FastMCP, CLI
libraries, and multimodal helper stacks are optional extras.

## Install

```bash
pip install thund3rbot
```

For local development:

```bash
pip install -e ".[dev]"
```

Useful extras:

```bash
pip install "thund3rbot[providers]"
pip install "thund3rbot[fastapi]"
pip install "thund3rbot[mcp]"
pip install "thund3rbot[all]"
```

Python 3.11+ is required.

## Minimal Agent

```python
import asyncio

from thund3rbot import AgentFramework, AgentScope, AgentSpec, FrameworkConfig, ModelConfig


async def main() -> None:
    framework = AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider="ollama", model="llama3.2")
        )
    )
    agent = framework.agent(
        AgentSpec(
            name="assistant",
            scope=AgentScope.TASK,
            instructions="Answer concisely using clear language.",
        )
    )

    result = await agent.run("Summarize the CAP theorem in two sentences.")
    print(result.output)


asyncio.run(main())
```

`core` imports are still available for compatibility, but new code should import
from `thund3rbot`.

## Runtime Model

Agents have separate identities for production use:

- `agent_id`: stable identity from the `AgentSpec`
- `session_id`: memory lane for conversation/history, defaulting to `agent_id`
- `run_id`: unique id for each call

Per-run context is passed explicitly and injected into the model message stream:

```python
result = await framework.run_agent(
    AgentSpec(name="support", scope=AgentScope.TASK, session_id="ticket-123"),
    "Draft a customer reply.",
    context={"customer_tier": "enterprise", "tone": "friendly"},
)
```

## Scopes

| Scope | Use |
| --- | --- |
| `task_agent` | Focused single-task execution. |
| `sub_agent` | Mid-level coordinator that can spawn task agents. |
| `orchestrator` | Top-level coordinator that can spawn sub-agents, including parallel sub-agent runs. |

Each scope receives a built-in scope contract before developer instructions and
skill instructions are added.

## Tools

Define tools with `@tool`, pass decorated callables directly to agents, or
register tools by name for reuse.

```python
from thund3rbot import AgentScope, tool


@tool(scopes=[AgentScope.TASK], risk="low", tags=["text"])
def word_count(text: str) -> int:
    """Return the word count of a string."""

    return len(text.split())


agent = framework.agent(
    AgentSpec(
        name="editor",
        scope=AgentScope.TASK,
        instructions="Edit text for clarity.",
        tools=[word_count],
    )
)
```

Registered tools can also be referenced by string name:

```python
framework.tools.register(word_count)
tools = ["word_count"]
```

Tool metadata supports `risk`, `requires_approval`, and `tags` for host-owned
automation policy.

## Approval Hooks

Use `RunOptions` to approve, reject, or modify tool calls before they execute.
This is useful for email, financial transactions, system administration, and
other irreversible work.

```python
from thund3rbot import RunOptions, ToolApproval


def approve_tool(context):
    if context.risk == "high":
        return ToolApproval(approved=False, reason="manual review required")
    return None


result = await agent.run(
    "Send the invoice reminder.",
    options=RunOptions(before_tool_call=approve_tool),
)
```

`after_tool_call` and `on_tool_error` hooks are also available for tracing,
auditing, and host-level recovery.

## Skills

Skills are reusable instruction/tool/context bundles. They do not create new
agents by themselves; they extend an agent's prompt and available tools.

```python
from thund3rbot import Skill

framework.skills.register(
    Skill(
        name="research",
        description="Careful research and synthesis",
        instructions="Check assumptions and cite uncertainty clearly.",
        tools=["web_search"],
        scopes={AgentScope.SUB_AGENT, AgentScope.ORCHESTRATOR},
    )
)
```

Markdown skills are supported:

```markdown
---
name: research
description: Careful research and synthesis
tools: [web_search]
requires: [citation_check]
scopes: [sub_agent, orchestrator]
---
Check assumptions and cite uncertainty clearly.
```

```python
framework.skills.load_dir("skills")
```

Circular skill dependencies raise `SkillConfigError`. Unknown Markdown tool
names raise `ToolNotFoundError` at load time.

## Multimodal Foundation

Agents can receive text or typed content parts. Tools can return typed artifacts.
The core only provides the data contracts; OCR, browser automation, media
processing, and vector stores belong in optional tools or host applications.

```python
from thund3rbot import Artifact, ContentPart


@framework.tools.register(scopes=[AgentScope.TASK])
def extract_page(url: str):
    """Extract page content."""

    return Artifact(type="markdown", uri=url, data="# Extracted page")


result = await framework.run_agent(
    AgentSpec(name="extractor", scope=AgentScope.TASK, tools=["extract_page"]),
    [
        ContentPart(type="text", text="Extract this page."),
        ContentPart(type="screenshot", uri="file:///tmp/page.png", mime_type="image/png"),
    ],
)
```

Supported content/artifact types are `text`, `image`, `audio`, `video`, `html`,
`markdown`, `json`, `file`, and `screenshot`.

## Typed Outputs

Agents can declare a Pydantic output schema. The framework asks the model for
JSON and deserializes `result.output`.

```python
from pydantic import BaseModel


class ResearchReport(BaseModel):
    summary: str
    sources: list[str]
    confidence: float


agent = framework.agent(
    AgentSpec(
        name="researcher",
        scope=AgentScope.SUB_AGENT,
        instructions="Research the topic and produce a structured report.",
        output_schema=ResearchReport,
    )
)
```

## Workflows

Workflows are in-process pipelines registered on a framework instance.

```python
@framework.workflow("brief")
async def brief(context, fw):
    result = await fw.step(
        "write",
        AgentSpec(name="writer", scope=AgentScope.TASK),
        f"Write a short brief about {context['topic']}.",
    )
    return {"brief": result.output}


result = await framework.workflows.run("brief", {"topic": "SQLite"})
```

## Adapters

FastAPI and MCP support are opt-in:

```python
pip install "thund3rbot[fastapi]"
pip install "thund3rbot[mcp]"
```

```python
from fastapi import FastAPI
from thund3rbot import AgentFramework, FrameworkConfig
from core.integrations.fastapi import create_agent_router

framework = AgentFramework(FrameworkConfig())
app = FastAPI()
app.include_router(create_agent_router(framework), prefix="/api/v1")
```

The compatibility adapter path is currently `core.integrations.*`.

## Model Configuration

For provider-backed models, install the provider extras and configure a model:

```python
from thund3rbot import AgentFramework, FrameworkConfig, ModelConfig, ProviderConfig

framework = AgentFramework(
    FrameworkConfig(
        default_model=ModelConfig(provider="openai", model="gpt-4o-mini"),
        providers={
            "openai": ProviderConfig(name="openai", api_key_env="OPENAI_API_KEY")
        },
    )
)
```

For tests or custom providers, pass a model factory:

```python
framework = AgentFramework(
    FrameworkConfig(
        default_model=ModelConfig(provider="custom", model="fake"),
        model_factory=lambda model_config: my_chat_model,
    )
)
```

## Project Layout

```text
core/
  framework.py             AgentFramework runtime
  agents.py                scoped task/sub/orchestrator agents
  tooling.py               tool registry and @tool decorator
  skills.py                Python and Markdown skill registry
  prompts.py               prompt registry and @prompt decorator
  memory.py                memory interfaces and in-memory store
  workflows.py             in-process workflow registry
  integrations/
    fastapi.py             optional FastAPI router adapter
    fastmcp.py             optional FastMCP/MCP helpers

thund3rbot/
  __init__.py              official public API facade
```
