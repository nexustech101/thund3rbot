# Agent Framework

An embeddable Python framework for building scoped agents and agent workflows with LangChain and LangGraph-compatible models. The framework is designed to be used inside your own applications the way you would embed an ML model: configure it, register domain tools and skills, create an agent, and call `run`.

FastAPI, FastMCP, and the CLI are host/application layers. They are useful examples, but the core framework does not depend on them during normal import or agent execution.

## Install

```bash
pip install -e .
```

Python 3.11+ is required.

## Minimal Agent

```python
from __future__ import annotations

# 1. Import necessary libraries and modules
import sys
import asyncio
from functools import lru_cache
from core import (
    AgentFramework, 
    AgentScope,  # StrEnum for agent scopes
    AgentSpec, 
    FrameworkConfig, 
    ModelConfig
)

# 2. Define a function to create an agent instance with caching to optimize performance
@lru_cache(maxsize=1)
def create_task_agent(
    *,
    provider: str, 
    model: str,
    name: str,
    scope: AgentScope,
    instructions: str
) -> AgentFramework.agent:
    """
    Create and return an instance of the AgentFramework with the 
    specified provider and model.

    Args:
        provider (str): The language model provider to use.
        model (str): The specific model to use from the provider.
        name (str): The name of the agent.
        scope (AgentScope): The scope of the agent.
        instructions (str): The instructions for the agent.

    Returns:
        AgentFramework.agent: An instance of the AgentFramework agent 
        configured with the specified provider and model.
    """
    try:
        # Configure the framework with the specified provider and model
        # This configuration is cached to avoid redundant setup on subsequent calls
        framework_config = AgentFramework(
            # Set the default model configuration for the agent
            FrameworkConfig(
                default_model=ModelConfig(provider=provider, model=model)
            )
        )

        # Create and return the agent instance using the configured framework
        return framework_config.agent(
            AgentSpec(
                name=name,
                scope=scope,
                instructions=instructions,
            )
        )
    except Exception as e:
        print(f"Error creating agent: {e}")
        raise


# 3. Define the main function to run the CLI tool
async def main() -> None:
    # Create an agent instance using the create_task_agent function
    agent = create_task_agent(
        provider="ollama",
        model="qwen3.5:9b",
        name="assistant",
        scope=AgentScope.TASK,
        instructions="Answer questions concisely using clear language."
    )

    # Run the agent in a loop to continuously accept user input
    # The agent will process the input and provide responses for the user
    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        result = await agent.run(user_input)
        print(result.output)


    # result = await agent.run("Summarize the CAP theorem in two sentences.")
    # print(result.output)


# 4. Run the main function when the script is executed
if __name__ == "__main__":
    asyncio.run(main())
```

## Scoped Agents

The framework has three scopes. Each scope has a built-in master prompt that defines the agent's duties before developer instructions and skill instructions are added.

| Scope | Use |
| --- | --- |
| `task_agent` | Focused single-task execution. |
| `sub_agent` | Mid-level coordinator that can spawn task agents. |
| `orchestrator` | Top-level coordinator that can spawn sub-agents, including parallel sub-agent runs. |

```python
agent = framework.agent(
    AgentSpec(
        name="researcher",
        scope=AgentScope.ORCHESTRATOR,
        instructions="Research, delegate when useful, and synthesize clearly.",
        tools=["web_search"],
        skills=["research"],
    )
)
```

## Tools

Core tools are framework-local. Define tools with `@tool`, pass decorated callables directly to agents, or register tools by name for reuse.

```python
from core import AgentScope, tool


@tool(scopes=[AgentScope.TASK, AgentScope.SUB_AGENT])
def normalize_title(title: str) -> str:
    """Normalize a title for display."""
    return " ".join(title.strip().split()).title()


@tool(scopes=[AgentScope.TASK])
def word_count(text: str) -> int:
    """Return the word count of a string."""
    return len(text.split())


agent = framework.agent(
    AgentSpec(
        name="editor",
        scope=AgentScope.TASK,
        instructions="Edit text for clarity and correctness.",
        tools=[normalize_title, word_count],
    )
)
```

String names still work for registered and MCP tools, and can be mixed with callables:

```python
tools=[normalize_title, "web_search"]
```

To register a reusable named tool:

```python
framework.tools.register(normalize_title)
```

Agents receive only the tools listed in their `AgentSpec` or granted by active skills. Scope grants restrict where a tool can be used.

## Prompts

Prompt templates can be registered locally. MCP prompts can also be loaded through the FastMCP integration.

```python
from core import prompt


@framework.prompts.register
@prompt(name="research_brief")
def research_brief_prompt(topic: str) -> str:
    return f"Research {topic}. Prefer primary sources."


text = framework.prompts.render("research_brief", topic="SQLite")
```

## Skills

Skills are reusable bundles of instructions and tool grants. They can be registered from Python:

```python
from core import Skill, tool


@tool(scopes=[AgentScope.SUB_AGENT])
async def search_sources(query: str) -> list[str]:
    """Search for credible sources on a topic."""
    ...

framework.skills.register(
    Skill(
        name="research",
        description="Careful research and synthesis",
        instructions="Check assumptions and cite uncertainty clearly.",
        tools=[search_sources, "web_search"],
        requires=["citation_check"],
        scopes=[AgentScope.SUB_AGENT, AgentScope.ORCHESTRATOR],
    )
)
```

Or loaded from Markdown files:

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

Circular skill dependencies raise `SkillConfigError`. Unknown Markdown tool names raise `ToolNotFoundError` at load time.

## Typed Outputs

Agents can declare a Pydantic output schema. When set, the framework asks the model for JSON and deserializes `result.output`.

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
        tools=["web_search"],
        skills=["research"],
        output_schema=ResearchReport,
    )
)

result = await agent.run("Impact of LLMs on software engineering.")
print(result.output.confidence)
```

## Run Options

Use `RunOptions` for per-call limits and step callbacks.

```python
from core import RunOptions

result = await agent.run(
    "Analyze this dataset.",
    options=RunOptions(
        max_steps=10,
        max_tool_calls=20,
        timeout_seconds=30,
        on_step=lambda step: print(f"[{step.index}] {step.tool}: {step.summary}"),
    ),
)

print(result.stop_reason)
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

`fw.step(name, spec, input)` is preferred inside workflows because it emits named workflow-step events. `fw.run_agent(...)` remains available for anonymous runs.

## Observability

Register one sync or async event hook for framework events:

```python
framework.on_event(lambda event: logger.info(event.model_dump()))
```

Events include agent starts/finishes, tool calls/results, and workflow step starts/finishes.

## FastAPI Adapter

FastAPI support is opt-in through an adapter.

```python
from fastapi import FastAPI
from core import AgentFramework, FrameworkConfig
from core.integrations.fastapi import create_agent_router

framework = AgentFramework(FrameworkConfig())

app = FastAPI()
app.include_router(create_agent_router(framework), prefix="/api/v1")
```

This exposes:

| Method | Path |
| --- | --- |
| `POST` | `/api/v1/agents/run` |
| `GET` | `/api/v1/agents/{run_id}` |
| `GET` | `/api/v1/agents/` |

## FastMCP Adapter

FastMCP support is also opt-in.

```python
from fastmcp import FastMCP
from core.integrations.fastmcp import register_fastmcp_tools

mcp = FastMCP("my-agent-tools")
register_fastmcp_tools(framework, mcp)
```

To load tools from a running MCP server under a namespace:

```python
await framework.tools.load_mcp(
    "http://127.0.0.1:8001/mcp/v1",
    namespace="crm",
    overrides={
        "crm.get_contact": "Fetch a CRM contact by email. Returns name, company, and deal stage.",
    },
)

agent = framework.agent(
    AgentSpec(
        name="sales_assistant",
        scope=AgentScope.TASK,
        tools=["crm.get_contact"],
    )
)
```

Grant a whole namespace with a wildcard:

```python
tools=["crm.*"]
```

To load MCP prompt metadata:

```python
await framework.prompts.load_mcp("http://127.0.0.1:8001/mcp/v1", namespace="crm")
```

## Model Configuration

The framework requires a config object. Environment variables are used only as provider conveniences when credentials are not supplied directly.

```python
from core import FrameworkConfig, ModelConfig, ProviderConfig

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
  tooling.py               generic tool registry
  skills.py                Python and Markdown skill registry
  workflows.py             in-process workflow registry
  integrations/
    fastapi.py             optional FastAPI router adapter
    fastmcp.py             optional FastMCP helpers

api/                       sample FastAPI host app
cli/                       sample CLI/console host app
tools/, workflows/, agents/ legacy/sample modules from the earlier attempt
```

`ml/` and `legacy/` are unrelated to the framework refactor.
