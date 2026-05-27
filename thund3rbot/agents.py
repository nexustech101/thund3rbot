"""Scoped agent implementations hidden behind the factory runtime."""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool as lc_tool

from thund3rbot.memory import BaseMemoryStore
from thund3rbot.parser import ResponseParser
from thund3rbot.types import (
    AgentFinished,
    AgentInput,
    AgentScope,
    AgentSpec,
    AgentStarted,
    AgentStatus,
    ModelConfig,
    RunOptions,
    RunResult,
    StepEvent,
    ToolApproval,
    ToolCalled,
    ToolCallContext,
    ToolResult,
    ToolSpec,
)

_TOOL_FORMAT_INSTRUCTIONS = """

When you need to call a tool, emit exactly one XML block:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

Wait for the tool result before producing the final answer.
"""

_STRUCTURED_OUTPUT_INSTRUCTIONS = """

Return the final answer as JSON that validates against this schema:
{schema}

Return only the JSON object. Do not wrap it in Markdown.
"""

_SCOPE_MASTER_PROMPTS: dict[AgentScope, str] = {
    AgentScope.TASK: (
        "Scope contract: You are a task agent. Own exactly one bounded task. "
        "Do not delegate. Use only the tools required to complete the task, then return the final result."
    ),
    AgentScope.SUB_AGENT: (
        "Scope contract: You are a sub-agent. Own a domain-specific subproblem inside a larger application. "
        "Use available tools directly, delegate atomic work to task agents when useful, and synthesize a local result."
    ),
    AgentScope.ORCHESTRATOR: (
        "Scope contract: You are an orchestrator. Decompose high-level goals, decide which sub-agents or tools "
        "are needed, coordinate their work, track completion, and synthesize the final application-level answer."
    ),
}


class BaseScopedAgent:
    """Base class for factory-owned scoped agents."""

    default_instructions = "You are a helpful agent."

    def __init__(self, factory: Any, spec: AgentSpec) -> None:
        self.factory = factory
        self.spec = spec
        self.name = spec.name
        self.scope = spec.scope
        self.agent_id = spec.agent_id
        self.memory: BaseMemoryStore = factory.memory

    async def run(
        self,
        input_text: AgentInput,
        context: dict[str, Any] | None = None,
        *,
        options: RunOptions | None = None,
    ) -> RunResult:
        options = options or RunOptions()
        session_id = self.spec.session_id or self.agent_id
        result = RunResult(
            agent_id=self.agent_id,
            session_id=session_id,
            name=self.name,
            scope=self.scope,
            status=AgentStatus.RUNNING,
            metadata={**self.spec.metadata},
        )
        self.factory.runs[result.run_id] = result
        started = time.perf_counter()
        await self.factory.emit(AgentStarted(run_id=result.run_id, agent_name=self.name, scope=self.scope, input=input_text))

        try:
            run_coro = self._run_details(input_text, context or {}, options, result)
            details = await asyncio.wait_for(run_coro, timeout=options.timeout_seconds) if options.timeout_seconds else await run_coro
            result.output = details["output"]
            result.stop_reason = details["stop_reason"]
            result.steps = details["steps"]
            result.tool_calls = details["tool_calls"]
            result.status = AgentStatus.COMPLETED if result.stop_reason != "error" else AgentStatus.FAILED
        except asyncio.TimeoutError:
            result.stop_reason = "timeout"
            result.status = AgentStatus.CANCELLED
        except Exception as exc:
            result.error = str(exc)
            result.stop_reason = "error"
            result.status = AgentStatus.FAILED
        finally:
            result.completed_at = datetime.now(UTC)
            duration_ms = (time.perf_counter() - started) * 1000
            await self.factory.emit(
                AgentFinished(
                    run_id=result.run_id,
                    agent_name=self.name,
                    output=result.output,
                    stop_reason=result.stop_reason,
                    steps=result.steps,
                    duration_ms=duration_ms,
                )
            )
        return result

    async def _run_details(
        self,
        input_text: AgentInput,
        context: dict[str, Any],
        options: RunOptions,
        result: RunResult,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def model_config(self) -> ModelConfig:
        return self.spec.model or self.factory.config.default_model

    @property
    def max_iterations(self) -> int:
        return self.spec.max_iterations or self.factory.config.max_iterations

    def _instructions(self, tools: list[BaseTool]) -> str:
        parts = [
            self.factory.scope_prompts.get(self.scope, _SCOPE_MASTER_PROMPTS[self.scope]),
            self.spec.instructions or self.default_instructions,
        ]
        for skill in self.factory.skills.resolve(self.spec.skills, scope=self.scope):
            if skill.instructions:
                parts.append(skill.instructions)
        instructions = "\n\n".join(part.strip() for part in parts if part.strip())
        if tools:
            instructions += _TOOL_FORMAT_INSTRUCTIONS
        if self.spec.output_schema is not None:
            instructions += _STRUCTURED_OUTPUT_INSTRUCTIONS.format(
                schema=json.dumps(self.spec.output_schema.model_json_schema(), indent=2)
            )
        return instructions

    def _tools(self) -> list[BaseTool]:
        return [spec.tool for spec in self._tool_specs()]

    def _tool_specs(self) -> list[ToolSpec]:
        refs = list(self.spec.tools)
        for skill in self.factory.skills.resolve(self.spec.skills, scope=self.scope):
            refs.extend(skill.tools)
        if not refs:
            return []
        return self.factory.tools.resolve(refs, scope=self.scope)

    def _transient_tool_spec(self, tool: BaseTool) -> ToolSpec:
        return ToolSpec(
            name=tool.name,
            public_name=tool.name,
            description=getattr(tool, "description", "") or "",
            tool=tool,
        )

    async def _invoke_model(self, model: Any, messages: list[Any]) -> Any:
        if hasattr(model, "ainvoke"):
            return await model.ainvoke(messages)
        if callable(model):
            response = model(messages)
            if inspect.isawaitable(response):
                return await response
            return response
        raise TypeError("Configured model does not support ainvoke() and is not callable.")

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        tool_specs: list[ToolSpec],
        *,
        run_id: str,
        session_id: str,
        step_index: int,
        options: RunOptions,
    ) -> ToolMessage:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        call_id = tool_call.get("id", name)
        selected_spec = next(
            (candidate for candidate in tool_specs if candidate.tool.name == name or candidate.name == name),
            None,
        )
        hook_context = ToolCallContext(
            run_id=run_id,
            agent_id=self.agent_id,
            session_id=session_id,
            agent_name=self.name,
            tool=name,
            arguments=args if isinstance(args, dict) else {},
            risk=selected_spec.risk if selected_spec else "low",
            requires_approval=selected_spec.requires_approval if selected_spec else False,
            metadata={
                "registered_name": selected_spec.name if selected_spec else name,
                "namespace": selected_spec.namespace if selected_spec else None,
                "tags": sorted(selected_spec.tags) if selected_spec else [],
            },
        )
        started = time.perf_counter()
        if selected_spec is None:
            output = f"Error: tool {name!r} is not available."
        else:
            approval = await _resolve_tool_approval(options, hook_context)
            if not approval.approved:
                output = f"Tool call rejected: {approval.reason or 'approval denied'}"
            else:
                if approval.arguments is not None:
                    args = approval.arguments
                    hook_context.arguments = args
                await self.factory.emit(ToolCalled(run_id=run_id, agent_name=self.name, tool=name, input=args))
                try:
                    output = await selected_spec.tool.ainvoke(args)
                    await _call_tool_output_hook(options, hook_context, output)
                    output = _stringify_tool_output(output)
                except Exception as exc:
                    await _call_tool_error_hook(options, hook_context, exc)
                    output = f"Tool error: {exc}"
        duration_ms = (time.perf_counter() - started) * 1000
        await self.factory.emit(ToolResult(run_id=run_id, agent_name=self.name, tool=name, output=output, duration_ms=duration_ms))
        await _call_step_callback(
            options,
            StepEvent(index=step_index, tool=name, input=args, output=output, summary=_summarize(output)),
        )
        return ToolMessage(content=output, tool_call_id=call_id)

    async def _tool_loop(
        self,
        input_text: AgentInput,
        context: dict[str, Any],
        tool_specs: list[ToolSpec],
        instructions: str,
        options: RunOptions,
        result: RunResult,
    ) -> dict[str, Any]:
        tools = [spec.tool for spec in tool_specs]
        model = self.factory.create_model(self.model_config)
        executable_model = model.bind_tools(tools) if tools and hasattr(model, "bind_tools") else model
        parser = ResponseParser({candidate.name for candidate in tools} if tools else None)
        history = await self.memory.get_history(result.session_id)
        messages: list[Any] = [SystemMessage(content=instructions)]
        if context:
            messages.append(SystemMessage(content=_format_context(context)))
        messages.extend(history)
        messages.append(HumanMessage(content=_input_content(input_text)))
        max_steps = options.max_steps or self.max_iterations
        max_tool_calls = options.max_tool_calls
        step_count = 0
        tool_count = 0
        partial_output = ""

        while step_count < max_steps:
            step_count += 1
            response = await self._invoke_model(executable_model, messages)
            if isinstance(response, str):
                response = AIMessage(content=response)

            parsed = parser.parse(response)
            tool_calls = getattr(response, "tool_calls", None) or parsed.langchain_tool_calls()
            if not tool_calls:
                output = _message_content(response) or parsed.response
                typed_output = self._coerce_output(output)
                await self.memory.add_message(result.session_id, HumanMessage(content=_input_content(input_text)))
                await self.memory.add_message(result.session_id, AIMessage(content=output))
                return {
                    "output": typed_output,
                    "stop_reason": "completed",
                    "steps": step_count,
                    "tool_calls": tool_count,
                }

            partial_output = parsed.response or partial_output
            if max_tool_calls is not None and tool_count + len(tool_calls) > max_tool_calls:
                return {
                    "output": self._coerce_output(partial_output) if partial_output else partial_output,
                    "stop_reason": "max_tool_calls",
                    "steps": step_count,
                    "tool_calls": tool_count,
                }

            messages.append(AIMessage(content=parsed.response, tool_calls=tool_calls))
            tool_messages = await asyncio.gather(
                *[
                    self._execute_tool_call(
                        call,
                        tool_specs,
                        run_id=result.run_id,
                        session_id=result.session_id,
                        step_index=step_count,
                        options=options,
                    )
                    for call in tool_calls
                ],
            )
            tool_count += len(tool_calls)
            messages.extend(tool_messages)

        return {
            "output": self._coerce_output(partial_output) if partial_output else partial_output,
            "stop_reason": "max_steps",
            "steps": step_count,
            "tool_calls": tool_count,
        }

    def _coerce_output(self, output: str) -> Any:
        if self.spec.output_schema is None:
            return output
        text = _strip_json_markdown(output)
        return self.spec.output_schema.model_validate_json(text)


class TaskAgent(BaseScopedAgent):
    default_instructions = "Complete the task precisely and return only the requested result."

    async def _run_details(
        self,
        input_text: AgentInput,
        context: dict[str, Any],
        options: RunOptions,
        result: RunResult,
    ) -> dict[str, Any]:
        tool_specs = self._tool_specs()
        tools = [spec.tool for spec in tool_specs]
        return await self._tool_loop(input_text, context, tool_specs, self._instructions(tools), options, result)


class SubAgent(TaskAgent):
    default_instructions = (
        "Coordinate the assigned subproblem. Use tools directly or delegate atomic work "
        "to task agents, then synthesize the local answer."
    )

    def _tools(self) -> list[BaseTool]:
        return [self._make_task_spawn_tool(), *super()._tools()]

    def _tool_specs(self) -> list[ToolSpec]:
        return [self._transient_tool_spec(self._make_task_spawn_tool()), *super()._tool_specs()]

    def _make_task_spawn_tool(self) -> BaseTool:
        parent = self

        @lc_tool
        async def create_task_agent(name: str, task: str, instructions: str = "") -> str:
            """Create and run a task agent for a focused subtask."""

            result = await parent.spawn_task_agent(name, task, instructions=instructions)
            return result.output or result.error or ""

        return create_task_agent  # type: ignore[return-value]

    async def spawn_task_agent(
        self,
        name: str,
        task: str,
        *,
        instructions: str = "",
        model: ModelConfig | None = None,
        tools: list[Any] | None = None,
        skills: list[str] | None = None,
        options: RunOptions | None = None,
    ) -> RunResult:
        spec = AgentSpec(
            name=name,
            session_id=self.spec.session_id,
            scope=AgentScope.TASK,
            instructions=instructions,
            model=model or self.spec.model,
            tools=tools if tools is not None else list(self.spec.tools),
            skills=skills if skills is not None else list(self.spec.skills),
            max_iterations=self.spec.max_iterations,
        )
        return await self.factory.run_agent(spec, task, options=options)


class OrchestratorAgent(SubAgent):
    default_instructions = (
        "Decompose the goal, choose the right sub-agents or tools, coordinate execution, "
        "and synthesize the final application-level result."
    )

    def _tools(self) -> list[BaseTool]:
        return [self._make_sub_spawn_tool(), self._make_parallel_sub_spawn_tool(), *TaskAgent._tools(self)]

    def _tool_specs(self) -> list[ToolSpec]:
        return [
            self._transient_tool_spec(self._make_sub_spawn_tool()),
            self._transient_tool_spec(self._make_parallel_sub_spawn_tool()),
            *TaskAgent._tool_specs(self),
        ]

    def _make_sub_spawn_tool(self) -> BaseTool:
        parent = self

        @lc_tool
        async def create_sub_agent(name: str, task: str, instructions: str = "") -> str:
            """Create and run a sub-agent for a coordinated subtask."""

            result = await parent.spawn_sub_agent(name, task, instructions=instructions)
            return result.output or result.error or ""

        return create_sub_agent  # type: ignore[return-value]

    def _make_parallel_sub_spawn_tool(self) -> BaseTool:
        parent = self

        @lc_tool
        async def create_sub_agents_parallel(tasks_json: str) -> str:
            """Run multiple sub-agents in parallel from a JSON array of task objects."""

            tasks = json.loads(tasks_json)
            results = await asyncio.gather(
                *[
                    parent.spawn_sub_agent(
                        item["name"],
                        item["task"],
                        instructions=item.get("instructions", item.get("system_prompt", "")),
                    )
                    for item in tasks
                ]
            )
            return "\n\n".join(f"[{result.name}] {result.output or result.error}" for result in results)

        return create_sub_agents_parallel  # type: ignore[return-value]

    async def spawn_sub_agent(
        self,
        name: str,
        task: str,
        *,
        instructions: str = "",
        model: ModelConfig | None = None,
        tools: list[Any] | None = None,
        skills: list[str] | None = None,
        options: RunOptions | None = None,
    ) -> RunResult:
        spec = AgentSpec(
            name=name,
            session_id=self.spec.session_id,
            scope=AgentScope.SUB_AGENT,
            instructions=instructions,
            model=model or self.spec.model,
            tools=tools if tools is not None else list(self.spec.tools),
            skills=skills if skills is not None else list(self.spec.skills),
            max_iterations=self.spec.max_iterations,
        )
        return await self.factory.run_agent(spec, task, options=options)


async def _call_step_callback(options: RunOptions, event: StepEvent) -> None:
    if options.on_step is None:
        return
    value = options.on_step(event)
    if inspect.isawaitable(value):
        await value


async def _resolve_tool_approval(options: RunOptions, context: ToolCallContext) -> ToolApproval:
    if options.before_tool_call is None:
        if context.requires_approval:
            return ToolApproval(approved=False, reason="tool requires approval")
        return ToolApproval()

    value = options.before_tool_call(context)
    if inspect.isawaitable(value):
        value = await value

    if value is None:
        return ToolApproval()
    if isinstance(value, ToolApproval):
        return value
    if isinstance(value, bool):
        return ToolApproval(approved=value)
    if isinstance(value, dict):
        if "approved" in value or "arguments" in value or "reason" in value:
            return ToolApproval.model_validate(value)
        return ToolApproval(arguments=value)
    raise TypeError("before_tool_call must return None, bool, dict, or ToolApproval.")


async def _call_tool_output_hook(options: RunOptions, context: ToolCallContext, output: Any) -> None:
    if options.after_tool_call is None:
        return
    value = options.after_tool_call(context, output)
    if inspect.isawaitable(value):
        await value


async def _call_tool_error_hook(options: RunOptions, context: ToolCallContext, exc: Exception) -> None:
    if options.on_tool_error is None:
        return
    value = options.on_tool_error(context, exc)
    if inspect.isawaitable(value):
        await value


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _input_content(value: AgentInput) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    content: list[dict[str, Any]] = []
    for part in value:
        item = {
            "type": part.type,
            "text": part.text,
            "data": part.data,
            "uri": part.uri,
            "mime_type": part.mime_type,
            "metadata": part.metadata,
        }
        content.append({key: item[key] for key in item if item[key] is not None and item[key] != {}})
    return content


def _format_context(context: dict[str, Any]) -> str:
    return "Runtime context:\n" + json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)


def _stringify_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if hasattr(output, "model_dump_json"):
        return output.model_dump_json()
    return str(output)


def _strip_json_markdown(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return stripped


def _summarize(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 160 else text[:157] + "..."
