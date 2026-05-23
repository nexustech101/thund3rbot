# Thund3rBot

Thund3rBot is a minimal Python framework for embedding scoped AI agents in real
software: APIs, backend services, scripts, CLIs, scrapers, automation jobs, and
internal tools.

It is designed for teams that want agentic workflows without handing their
application logic to a hosted agent platform. You provide the model, tools,
instructions, context, memory, and host application. Thund3rBot provides the
small runtime kernel that keeps those pieces organized.

## Why Thund3rBot?

Most agent frameworks either become application platforms or thin wrappers around
one provider. Thund3rBot takes a smaller stance:

- **Embeddable by default**: create a runtime, register tools, create an agent,
  call `run`.
- **Scoped agents**: choose between task agents, sub-agents, and orchestrators.
- **Runtime-local registries**: tools, skills, prompts, workflows, runs, and
  memory are owned by an `AgentFramework` instance.
- **Provider-flexible**: use local/open-source models through LangChain-compatible
  chat models, or configure hosted providers through optional extras.
- **Production-shaped controls**: unique run IDs, session memory, context
  injection, structured outputs, tool budgets, timeouts, events, and approval
  hooks.
- **Lean package surface**: FastAPI, MCP, provider SDKs, CLI helpers, and other
  heavier dependencies are optional.

Good fits include webpage extraction, email drafting, workflow automation,
sentiment analysis, structured data extraction, system administration helpers,
recurring business tasks, and embedding model-backed behavior in service routes.

## Installation

```bash
pip install thund3rbot
```

The base install includes only the runtime essentials.

Optional extras:

```bash
pip install "thund3rbot[providers]"  # OpenAI, Anthropic, Google, Ollama adapters
pip install "thund3rbot[fastapi]"    # FastAPI router adapter
pip install "thund3rbot[mcp]"        # MCP/FastMCP helpers
pip install "thund3rbot[all]"        # everything
```

For local development:

```bash
pip install -e ".[dev]"
```

Python 3.11+ is required.

## Quickstart

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
            instructions="Answer concisely in plain English.",
        )
    )

    result = await agent.run("Summarize the CAP theorem in two sentences.")
    print(result.output)


asyncio.run(main())
```

All public imports come from `thund3rbot`.

## Core Concepts

### Framework Runtime

`AgentFramework` is the runtime boundary. It owns local registries for tools,
skills, prompts, workflows, events, memory, and run results.

```python
from thund3rbot import AgentFramework, FrameworkConfig

framework = AgentFramework(FrameworkConfig())
```

### Agent Identity

Thund3rBot separates identity for production use:

- `agent_id`: stable identity from an `AgentSpec`
- `session_id`: memory lane for conversation history, defaulting to `agent_id`
- `run_id`: unique ID for each call

```python
result = await framework.run_agent(
    AgentSpec(name="support", scope=AgentScope.TASK, session_id="ticket-123"),
    "Draft a customer reply.",
    context={"customer_tier": "enterprise", "tone": "friendly"},
)
```

Per-run `context` is injected into the model message stream so host applications
can pass request, account, workflow, or policy state without folding it into the
user prompt.

### Agent Scopes

| Scope | Use |
| --- | --- |
| `task_agent` | Focused single-task execution. |
| `sub_agent` | Coordinates a domain-specific subproblem and can spawn task agents. |
| `orchestrator` | Decomposes larger work and can coordinate sub-agents. |

Each scope receives a built-in scope contract before developer instructions and
skill instructions are added.

## Tools

Tools are plain Python callables or LangChain-compatible tools. They can be
registered globally on a framework instance, passed directly to an agent, or
granted by skills.

```python
from thund3rbot import AgentScope, AgentSpec, tool


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

Reusable named tools:

```python
framework.tools.register(word_count)

agent = framework.agent(
    AgentSpec(name="counter", scope=AgentScope.TASK, tools=["word_count"])
)
```

Tool metadata supports `risk`, `requires_approval`, and `tags`, which host
applications can use to enforce their own safety and review policies.

## Approval Hooks

Use `RunOptions` to approve, reject, or modify tool calls before execution.
This is intentionally small and host-owned: Thund3rBot exposes the hook; your
application decides the policy.

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

`RunOptions` also supports step callbacks, tool budgets, timeouts,
`after_tool_call`, and `on_tool_error`.

## Skills

Skills are reusable bundles of instructions and tool grants. They do not create
agents by themselves; they extend an agent's prompt and available tool set.

```python
from thund3rbot import AgentScope, Skill

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

## Structured Outputs

Agents can declare a Pydantic output schema. The runtime asks the model for JSON
and deserializes `result.output`.

```python
from pydantic import BaseModel
from thund3rbot import AgentScope, AgentSpec


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

## Multimodal Inputs and Artifacts

Thund3rBot includes lightweight data contracts for multimodal work. The base
package does not ship OCR, browser automation, media processing, vector stores,
or document parsers. Those belong in optional tools or host applications.

```python
from thund3rbot import AgentScope, AgentSpec, Artifact, ContentPart


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

Supported content and artifact types are `text`, `image`, `audio`, `video`,
`html`, `markdown`, `json`, `file`, and `screenshot`.

## Workflows

Workflows are in-process pipelines registered on a framework instance. Use
`fw.step(...)` inside workflows when you want named workflow-step events.

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

## FastAPI Adapter

FastAPI support is optional.

```bash
pip install "thund3rbot[fastapi]"
```

```python
from fastapi import FastAPI
from thund3rbot import AgentFramework, FrameworkConfig
from thund3rbot.integrations.fastapi import create_agent_router

framework = AgentFramework(FrameworkConfig())
app = FastAPI()
app.include_router(create_agent_router(framework), prefix="/api/v1")
```

The router exposes:

| Method | Path |
| --- | --- |
| `POST` | `/api/v1/agents/run` |
| `GET` | `/api/v1/agents/{run_id}` |
| `GET` | `/api/v1/agents/` |

## MCP and FastMCP

MCP support is optional.

```bash
pip install "thund3rbot[mcp]"
```

```python
from fastmcp import FastMCP
from thund3rbot.integrations.fastmcp import register_fastmcp_tools

mcp = FastMCP("my-agent-tools")
register_fastmcp_tools(framework, mcp)
```

Tools and prompt metadata can also be loaded from MCP servers under namespaces:

```python
await framework.tools.load_mcp(
    "http://127.0.0.1:8001/mcp/v1",
    namespace="crm",
    overrides={
        "crm.get_contact": "Fetch a CRM contact by email.",
    },
)

await framework.prompts.load_mcp(
    "http://127.0.0.1:8001/mcp/v1",
    namespace="crm",
)
```

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

For tests, local wrappers, or custom providers, pass a model factory:

```python
framework = AgentFramework(
    FrameworkConfig(
        default_model=ModelConfig(provider="custom", model="fake"),
        model_factory=lambda model_config: my_chat_model,
    )
)
```

The custom model should provide `ainvoke(messages)`, or be a callable that
returns a response. If it supports `bind_tools(tools)`, Thund3rBot will call it
when tools are available.

## Public API

Common imports:

```python
from thund3rbot import (
    AgentFramework,
    AgentScope,
    AgentSpec,
    Artifact,
    ContentPart,
    FrameworkConfig,
    ModelConfig,
    ProviderConfig,
    RunOptions,
    Skill,
    ToolApproval,
    prompt,
    tool,
)
```

Optional adapters:

```python
from thund3rbot.integrations.fastapi import create_agent_router
from thund3rbot.integrations.fastmcp import register_fastmcp_tools
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

FastAPI-dependent tests are skipped unless the `fastapi` extra is installed.

## Project Layout

```text
thund3rbot/
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
  __init__.py              public API
```

## License

MIT License. See [LICENSE](LICENSE).
