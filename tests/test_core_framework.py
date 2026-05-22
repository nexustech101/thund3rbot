from __future__ import annotations

import subprocess
import sys

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class FakeChatModel:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages):
        return AIMessage(content=self.responses.pop(0))


def framework_with_fake(*responses: str):
    from core import AgentFramework, FrameworkConfig, ModelConfig

    return AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider="ollama", model="fake"),
            model_factory=lambda _: FakeChatModel(*responses),
        )
    )


def test_import_core_does_not_import_host_frameworks():
    script = (
        "import core, sys; "
        "blocked=[n for n in ('fastapi','fastmcp','click','rich') if n in sys.modules]; "
        "raise SystemExit(1 if blocked else 0)"
    )
    completed = subprocess.run([sys.executable, "-c", script], check=False)
    assert completed.returncode == 0


def test_config_accepts_model_and_model_name_alias():
    from core import ModelConfig

    assert ModelConfig(provider="openai", model="gpt-4o-mini").model == "gpt-4o-mini"
    assert ModelConfig(provider="openai", model_name="gpt-4o-mini").model == "gpt-4o-mini"
    assert ModelConfig(provider="openai", model_name="gpt-4o-mini").model_name == "gpt-4o-mini"


def test_tool_registration_and_scope_visibility():
    from core import AgentFramework, AgentScope, FrameworkConfig

    framework = AgentFramework(FrameworkConfig(enable_default_tools=False))

    @framework.tools.register(scopes=[AgentScope.TASK])
    def shout(text: str) -> str:
        """Uppercase text."""

        return text.upper()

    assert "shout" in framework.tools.names()
    assert [tool.name for tool in framework.tools.get(scope=AgentScope.TASK)] == ["shout"]
    assert framework.tools.get(scope=AgentScope.ORCHESTRATOR) == []


def test_python_and_markdown_skills(tmp_path):
    from core import Skill, ToolNotFoundError

    framework = framework_with_fake("done")
    framework.skills.register(Skill(name="citation_check", instructions="Check citations."))
    framework.skills.register(
        Skill(name="python-skill", instructions="Use Python style.", requires=["citation_check"])
    )

    (tmp_path / "research.md").write_text(
        (
            "---\nname: research\ndescription: Research helper\ntools: [echo]\n"
            "requires: [citation_check]\nscopes: [task_agent]\n---\nBe precise."
        ),
        encoding="utf-8",
    )
    loaded = framework.skills.load_dir(tmp_path)

    assert framework.skills.get("python-skill").instructions == "Use Python style."
    assert [skill.name for skill in framework.skills.resolve(["research"])] == ["citation_check", "research"]
    assert loaded[0].name == "research"
    assert loaded[0].tools == ["echo"]

    (tmp_path / "bad.md").write_text("---\nname: bad\ntools: [missing]\n---\nNope.", encoding="utf-8")
    with pytest.raises(ToolNotFoundError):
        framework.skills.load_dir(tmp_path)


def test_skill_cycle_raises_at_registration():
    from core import Skill, SkillConfigError

    framework = framework_with_fake("done")
    framework.skills.register(Skill(name="a", requires=["b"]))
    with pytest.raises(SkillConfigError):
        framework.skills.register(Skill(name="b", requires=["a"]))


@pytest.mark.asyncio
async def test_task_agent_runs_with_fake_model():
    from core import AgentScope, AgentSpec

    framework = framework_with_fake("hello")
    result = await framework.run_agent(
        AgentSpec(name="assistant", scope=AgentScope.TASK),
        "Say hello",
    )

    assert result.status == "completed"
    assert result.output == "hello"


@pytest.mark.asyncio
async def test_task_agent_tool_loop_uses_registered_callable():
    from core import AgentScope, AgentSpec

    framework = framework_with_fake(
        '<tool_call>{"name": "double", "arguments": {"value": 4}}</tool_call>',
        "The answer is 8.",
    )

    @framework.tools.register(scopes=[AgentScope.TASK])
    def double(value: int) -> int:
        """Double a value."""

        return value * 2

    result = await framework.run_agent(
        AgentSpec(name="calculator", scope=AgentScope.TASK, tools=["double"]),
        "Double 4",
    )

    assert result.status == "completed"
    assert result.output == "The answer is 8."


@pytest.mark.asyncio
async def test_agent_spec_accepts_decorated_tool_callable_directly():
    from core import AgentScope, AgentSpec, tool

    framework = framework_with_fake(
        '<tool_call>{"name": "word_count", "arguments": {"text": "one two three"}}</tool_call>',
        "There are 3 words.",
    )

    @tool(scopes=[AgentScope.TASK])
    def word_count(text: str) -> int:
        """Return the number of words."""

        return len(text.split())

    result = await framework.run_agent(
        AgentSpec(name="counter", scope=AgentScope.TASK, tools=[word_count]),
        "Count the words.",
    )

    assert result.output == "There are 3 words."
    assert result.tool_calls == 1


@pytest.mark.asyncio
async def test_typed_output_schema_returns_model_instance():
    from core import AgentScope, AgentSpec

    class ResearchReport(BaseModel):
        summary: str
        sources: list[str]
        confidence: float

    framework = framework_with_fake('{"summary": "Useful", "sources": ["a"], "confidence": 0.8}')
    result = await framework.run_agent(
        AgentSpec(
            name="researcher",
            scope=AgentScope.TASK,
            output_schema=ResearchReport,
        ),
        "Return JSON.",
    )

    assert isinstance(result.output, ResearchReport)
    assert result.output.confidence == 0.8


@pytest.mark.asyncio
async def test_run_options_step_callback_and_tool_budget():
    from core import AgentScope, AgentSpec, RunOptions

    framework = framework_with_fake(
        '<tool_call>{"name": "double", "arguments": {"value": 4}}</tool_call>',
        "done",
    )
    events = []

    @framework.tools.register(scopes=[AgentScope.TASK])
    def double(value: int) -> int:
        """Double a value."""

        return value * 2

    result = await framework.run_agent(
        AgentSpec(name="calculator", scope=AgentScope.TASK, tools=["double"]),
        "Double 4",
        options=RunOptions(max_steps=3, max_tool_calls=1, on_step=lambda event: events.append(event)),
    )

    assert result.stop_reason == "completed"
    assert result.tool_calls == 1
    assert events[0].tool == "double"


@pytest.mark.asyncio
async def test_run_options_max_tool_calls_stops_with_partial_result():
    from core import AgentScope, AgentSpec, RunOptions

    framework = framework_with_fake(
        '<tool_call>{"name": "double", "arguments": {"value": 4}}</tool_call>',
    )

    @framework.tools.register(scopes=[AgentScope.TASK])
    def double(value: int) -> int:
        """Double a value."""

        return value * 2

    result = await framework.run_agent(
        AgentSpec(name="calculator", scope=AgentScope.TASK, tools=["double"]),
        "Double 4",
        options=RunOptions(max_tool_calls=0),
    )

    assert result.stop_reason == "max_tool_calls"
    assert result.tool_calls == 0


@pytest.mark.asyncio
async def test_sub_agent_and_orchestrator_spawn_children_on_same_runtime():
    from core import AgentScope, AgentSpec

    framework = framework_with_fake("child done")
    sub_agent = framework.agent(AgentSpec(name="sub", scope=AgentScope.SUB_AGENT))
    sub_result = await sub_agent.spawn_task_agent("child-task", "Do child task")

    framework.config.model_factory = lambda _: FakeChatModel("sub done")
    orchestrator = framework.agent(AgentSpec(name="orch", scope=AgentScope.ORCHESTRATOR))
    orch_result = await orchestrator.spawn_sub_agent("child-sub", "Do sub task")

    assert sub_result.output == "child done"
    assert orch_result.output == "sub done"
    assert sub_result.run_id in framework.runs
    assert orch_result.run_id in framework.runs


@pytest.mark.asyncio
async def test_workflow_registration_and_run():
    framework = framework_with_fake("unused")

    @framework.workflow("collect", description="Collect data")
    async def collect(context, fw):
        return {"value": context["value"], "tools": fw.tools.names()}

    result = await framework.workflows.run("collect", {"value": 42})

    assert result.status == "completed"
    assert result.output["value"] == 42
    assert "echo" in result.output["tools"]


@pytest.mark.asyncio
async def test_framework_events_and_named_workflow_steps():
    from core import AgentScope, AgentSpec

    framework = framework_with_fake("step output")
    events = []
    framework.on_event(lambda event: events.append(event.event_type))

    @framework.workflow("research_brief")
    async def research_brief(context, fw):
        result = await fw.step(
            "research",
            AgentSpec(name="researcher", scope=AgentScope.TASK),
            context["query"],
        )
        return {"result": result.output}

    result = await framework.workflows.run("research_brief", {"query": "SQLite"})

    assert result.output == {"result": "step output"}
    assert "workflow_step_started" in events
    assert "agent_started" in events
    assert "agent_finished" in events
    assert "workflow_step_finished" in events


def test_prompt_registry_and_decorator():
    from core import prompt

    framework = framework_with_fake("unused")

    @framework.prompts.register
    @prompt(name="brief")
    def brief_prompt(topic: str) -> str:
        return f"Brief: {topic}"

    assert framework.prompts.render("brief", topic="SQLite") == "Brief: SQLite"


def test_namespaced_tool_resolution_uses_public_tool_name():
    framework = framework_with_fake("unused")

    def get_contact(email: str) -> str:
        return email

    lc_tool = StructuredTool.from_function(get_contact, name="get_contact", description="Get contact")
    framework.tools.add(
        lc_tool,
        name="crm.get_contact",
        public_name="get_contact",
        namespace="crm",
        description="Fetch a CRM contact by email.",
    )

    resolved = framework.tools.get(["crm.*"])

    assert framework.tools.get_spec("crm.get_contact").description == "Fetch a CRM contact by email."
    assert resolved[0].name == "get_contact"


def test_fastapi_adapter_can_mount_router_without_core_importing_fastapi():
    from fastapi import FastAPI

    from core.integrations.fastapi import create_agent_router

    framework = framework_with_fake("ok")
    app = FastAPI()
    app.include_router(create_agent_router(framework), prefix="/api/v1")

    routes = {route.path for route in app.routes}
    assert "/api/v1/agents/run" in routes


def test_fastmcp_adapter_import_does_not_import_fastmcp():
    sys.modules.pop("fastmcp", None)

    import core.integrations.fastmcp  # noqa: F401

    assert "fastmcp" not in sys.modules
