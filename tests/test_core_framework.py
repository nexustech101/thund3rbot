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


def test_public_imports_from_thund3rbot_and_core_compatibility():
    from core import AgentSpec as LegacyAgentSpec
    from thund3rbot import AgentFramework, AgentScope, AgentSpec, FrameworkConfig, tool

    framework = AgentFramework(FrameworkConfig(enable_default_tools=False))

    @tool(scopes=[AgentScope.TASK])
    def ping() -> str:
        """Ping."""

        return "pong"

    assert AgentSpec is not None
    assert LegacyAgentSpec is AgentSpec
    assert framework.tools.register(ping) is ping


def test_import_thund3rbot_does_not_import_optional_frameworks_or_providers():
    script = (
        "import thund3rbot, sys; "
        "blocked=[n for n in ("
        "'fastapi','fastmcp','mcp','click','rich','langchain_openai',"
        "'langchain_ollama','langchain_anthropic','langchain_google_genai'"
        ") if n in sys.modules]; "
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
    from core import AgentFramework, AgentScope, FrameworkConfig, tool

    framework = AgentFramework(FrameworkConfig(enable_default_tools=False))

    @tool(scopes=[AgentScope.TASK], risk="high", requires_approval=True, tags=["external"])
    def shout(text: str) -> str:
        """Uppercase text."""

        return text.upper()

    framework.tools.register(shout)

    assert "shout" in framework.tools.names()
    spec = framework.tools.get_spec("shout")
    assert spec.risk == "high"
    assert spec.requires_approval is True
    assert spec.tags == {"external"}
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
async def test_repeated_runs_have_unique_run_ids_and_stable_session_memory():
    from core import AgentFramework, AgentScope, AgentSpec, FrameworkConfig, ModelConfig

    class CapturingModel:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content=f"ok-{len(self.calls)}")

    model = CapturingModel()
    framework = AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider="ollama", model="fake"),
            model_factory=lambda _: model,
        )
    )
    agent = framework.agent(
        AgentSpec(name="assistant", scope=AgentScope.TASK, session_id="thread-1")
    )

    first = await agent.run("first")
    second = await agent.run("second")

    assert first.run_id != second.run_id
    assert first.agent_id == second.agent_id == agent.agent_id
    assert first.session_id == second.session_id == "thread-1"
    assert first.run_id in framework.runs
    assert second.run_id in framework.runs
    assert any(getattr(message, "content", "") == "first" for message in model.calls[1])
    assert any(getattr(message, "content", "") == "ok-1" for message in model.calls[1])


@pytest.mark.asyncio
async def test_context_is_injected_into_model_messages():
    from core import AgentFramework, AgentScope, AgentSpec, FrameworkConfig, ModelConfig

    captured = []

    class CapturingModel:
        async def ainvoke(self, messages):
            captured.extend(messages)
            return AIMessage(content="ok")

    framework = AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider="ollama", model="fake"),
            model_factory=lambda _: CapturingModel(),
        )
    )

    await framework.run_agent(
        AgentSpec(name="assistant", scope=AgentScope.TASK),
        "Use context",
        context={"account_id": "acct_123", "mode": "dry_run"},
    )

    contents = [getattr(message, "content", "") for message in captured]
    assert any("Runtime context:" in value and "acct_123" in value for value in contents)


@pytest.mark.asyncio
async def test_timeout_is_not_successful_completion():
    import asyncio

    from core import AgentScope, AgentSpec, RunOptions

    class SlowModel:
        async def ainvoke(self, messages):
            await asyncio.sleep(0.05)
            return AIMessage(content="late")

    from core import AgentFramework, FrameworkConfig, ModelConfig

    framework = AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider="ollama", model="fake"),
            model_factory=lambda _: SlowModel(),
        )
    )

    result = await framework.run_agent(
        AgentSpec(name="assistant", scope=AgentScope.TASK),
        "Wait",
        options=RunOptions(timeout_seconds=0.001),
    )

    assert result.stop_reason == "timeout"
    assert result.status == "cancelled"


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
async def test_tool_approval_hook_can_modify_arguments_and_observe_output():
    from core import AgentScope, AgentSpec, RunOptions, tool

    framework = framework_with_fake(
        '<tool_call>{"name": "double", "arguments": {"value": 4}}</tool_call>',
        "done",
    )
    seen = []

    @framework.tools.register
    @tool(scopes=[AgentScope.TASK], risk="medium", requires_approval=True)
    def double(value: int) -> int:
        """Double a value."""

        return value * 2

    def before(context):
        seen.append(("before", context.risk, context.requires_approval, context.arguments))
        return {"arguments": {"value": 5}}

    def after(context, output):
        seen.append(("after", context.arguments, output))

    result = await framework.run_agent(
        AgentSpec(name="calculator", scope=AgentScope.TASK, tools=["double"]),
        "Double 4",
        options=RunOptions(before_tool_call=before, after_tool_call=after),
    )

    assert result.output == "done"
    assert seen[0] == ("before", "medium", True, {"value": 4})
    assert seen[1] == ("after", {"value": 5}, 10)


@pytest.mark.asyncio
async def test_tool_approval_hook_can_reject_tool_call():
    from core import AgentScope, AgentSpec, RunOptions, tool

    framework = framework_with_fake(
        '<tool_call>{"name": "send_email", "arguments": {"to": "a@example.com"}}</tool_call>',
        "not sent",
    )
    called = False

    @framework.tools.register
    @tool(scopes=[AgentScope.TASK], risk="high", requires_approval=True)
    def send_email(to: str) -> str:
        """Send email."""

        nonlocal called
        called = True
        return to

    result = await framework.run_agent(
        AgentSpec(name="mailer", scope=AgentScope.TASK, tools=["send_email"]),
        "Send email",
        options=RunOptions(before_tool_call=lambda context: False),
    )

    assert result.output == "not sent"
    assert called is False


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
async def test_sub_agent_llm_can_use_spawn_tool():
    from core import AgentFramework, AgentScope, AgentSpec, FrameworkConfig, ModelConfig

    parent_model = FakeChatModel(
        '<tool_call>{"name": "create_task_agent", "arguments": {"name": "child", "task": "Do it"}}</tool_call>',
        "parent done",
    )
    child_model = FakeChatModel("child done")
    models = [parent_model, child_model]
    framework = AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider="ollama", model="fake"),
            model_factory=lambda _: models.pop(0),
        )
    )

    result = await framework.run_agent(
        AgentSpec(name="sub", scope=AgentScope.SUB_AGENT),
        "Delegate this",
    )

    assert result.output == "parent done"
    assert result.tool_calls == 1
    assert len(framework.runs) == 2


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


@pytest.mark.asyncio
async def test_multimodal_content_and_artifact_tool_flow():
    from core import AgentFramework, AgentScope, AgentSpec, Artifact, ContentPart, FrameworkConfig, ModelConfig

    captured_messages = []

    class CaptureModel:
        def __init__(self):
            self.responses = [
                '<tool_call>{"name": "extract_page", "arguments": {"url": "https://example.com"}}</tool_call>',
                "done",
            ]

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            captured_messages.append(messages)
            return AIMessage(content=self.responses.pop(0))

    artifacts = []

    framework = AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider="ollama", model="fake"),
            model_factory=lambda _: CaptureModel(),
        )
    )

    @framework.tools.register(scopes=[AgentScope.TASK])
    def extract_page(url: str):
        """Extract a page."""

        artifact = Artifact(type="markdown", data="# Example", uri=url)
        artifacts.append(artifact)
        return artifact

    result = await framework.run_agent(
        AgentSpec(name="extractor", scope=AgentScope.TASK, tools=["extract_page"]),
        [
            ContentPart(type="text", text="Extract this page."),
            ContentPart(type="screenshot", uri="file:///tmp/page.png", mime_type="image/png"),
        ],
    )

    human_content = captured_messages[0][1].content
    assert human_content[0]["type"] == "text"
    assert human_content[1]["type"] == "screenshot"
    assert artifacts[0].model_dump()["type"] == "markdown"
    assert result.output == "done"


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
