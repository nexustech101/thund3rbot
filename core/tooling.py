"""Generic framework tool registry and public ``@tool`` decorator."""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from core.types import AgentScope, ToolNotFoundError, ToolRef, ToolSpec


@dataclass(frozen=True)
class ToolMetadata:
    scopes: tuple[AgentScope, ...] | None = None
    name: str | None = None
    description: str | None = None


def tool(
    func: Callable[..., Any] | None = None,
    *,
    scopes: Iterable[AgentScope | str] | None = None,
    name: str | None = None,
    description: str | None = None,
):
    """Decorate a Python function as an agent tool without registering it globally."""

    normalised = None if scopes is None else tuple(_normalise_scopes(scopes))

    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, "__agent_tool_metadata__", ToolMetadata(normalised, name, description))
        return fn

    if func is None:
        return _decorate
    return _decorate(func)


def _normalise_scopes(scopes: Iterable[AgentScope | str] | None) -> set[AgentScope]:
    if scopes is None:
        return set()
    return {scope if isinstance(scope, AgentScope) else AgentScope(scope) for scope in scopes}


class ToolRegistry:
    """Runtime-local registry for LangChain tools and plain Python callables."""

    def __init__(self, framework: Any | None = None) -> None:
        self._framework = framework
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        tool_or_func: BaseTool | Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        scopes: Iterable[AgentScope | str] | None = None,
    ):
        """Register a tool directly or use as a decorator."""

        def _register(obj: BaseTool | Callable[..., Any]):
            metadata: ToolMetadata | None = getattr(obj, "__agent_tool_metadata__", None)
            effective_name = name or (metadata.name if metadata else None)
            effective_description = description or (metadata.description if metadata else None)
            effective_scopes = _normalise_scopes(scopes)
            if not effective_scopes and metadata and metadata.scopes is not None:
                effective_scopes = set(metadata.scopes)

            lc_tool = self._coerce_tool(obj, name=effective_name, description=effective_description)
            spec = ToolSpec(
                name=lc_tool.name,
                public_name=lc_tool.name,
                description=effective_description or getattr(lc_tool, "description", "") or "",
                scopes=effective_scopes,
                tool=lc_tool,
                source=None if isinstance(obj, BaseTool) else obj,
            )
            self._tools[spec.name] = spec
            return obj

        if tool_or_func is None:
            return _register
        return _register(tool_or_func)

    def add(
        self,
        lc_tool: BaseTool,
        *,
        scopes: Iterable[AgentScope | str] | None = None,
        name: str | None = None,
        public_name: str | None = None,
        description: str | None = None,
        namespace: str | None = None,
    ) -> BaseTool:
        spec_name = name or lc_tool.name
        exposed_tool = self._clone_tool(lc_tool, public_name or lc_tool.name, description)
        self._tools[spec_name] = ToolSpec(
            name=spec_name,
            public_name=exposed_tool.name,
            description=description or getattr(exposed_tool, "description", "") or "",
            scopes=_normalise_scopes(scopes),
            tool=exposed_tool,
            source=None,
            namespace=namespace,
        )
        return exposed_tool

    def get(
        self,
        refs: Iterable[ToolRef] | None = None,
        *,
        scope: AgentScope | str | None = None,
    ) -> list[BaseTool]:
        return [spec.tool for spec in self.resolve(refs, scope=scope)]

    def resolve(
        self,
        refs: Iterable[ToolRef] | None = None,
        *,
        scope: AgentScope | str | None = None,
    ) -> list[ToolSpec]:
        scope_value = scope if isinstance(scope, AgentScope) or scope is None else AgentScope(scope)
        selected = list(refs) if refs is not None else list(self._tools)
        result: list[ToolSpec] = []

        for ref in selected:
            if isinstance(ref, str):
                result.extend(self._resolve_string(ref, scope_value))
                continue
            if isinstance(ref, BaseTool):
                result.append(self._spec_from_direct(ref, scope_value))
                continue
            if callable(ref):
                result.append(self._spec_from_direct(ref, scope_value))
                continue
            raise TypeError(f"Unsupported tool reference: {ref!r}")

        return _dedupe_specs(result)

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def load_mcp(
        self,
        url: str,
        *,
        namespace: str,
        names: Iterable[str] | None = None,
        scopes: Iterable[AgentScope | str] | None = None,
        overrides: dict[str, str] | None = None,
    ) -> list[BaseTool]:
        """Load MCP tools under a namespace without importing MCP until called."""

        if self._framework is None:
            raise RuntimeError("MCP loading requires a ToolRegistry attached to an AgentFramework.")
        from core.integrations.fastmcp import load_mcp_tools

        return await load_mcp_tools(
            self._framework,
            url,
            namespace=namespace,
            names=names,
            scopes=scopes,
            overrides=overrides,
        )

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def _resolve_string(self, ref: str, scope: AgentScope | None) -> list[ToolSpec]:
        if ref.endswith(".*"):
            prefix = ref[:-1]
            matches = [spec for name, spec in self._tools.items() if name.startswith(prefix)]
            if not matches:
                raise ToolNotFoundError(f"No tools found for wildcard {ref!r}.")
            return [spec for spec in matches if self._allowed_for_scope(spec, scope)]

        spec = self._tools.get(ref)
        if spec is None:
            raise ToolNotFoundError(f"Tool {ref!r} is not registered.")
        if not self._allowed_for_scope(spec, scope):
            return []
        return [spec]

    def _spec_from_direct(self, ref: BaseTool | Callable[..., Any], scope: AgentScope | None) -> ToolSpec:
        metadata: ToolMetadata | None = getattr(ref, "__agent_tool_metadata__", None)
        scopes = set(metadata.scopes or ()) if metadata and metadata.scopes is not None else set()
        lc_tool = self._coerce_tool(
            ref,
            name=metadata.name if metadata else None,
            description=metadata.description if metadata else None,
        )
        spec = ToolSpec(
            name=lc_tool.name,
            public_name=lc_tool.name,
            description=getattr(lc_tool, "description", "") or "",
            scopes=scopes,
            tool=lc_tool,
            source=None if isinstance(ref, BaseTool) else ref,
        )
        if not self._allowed_for_scope(spec, scope):
            raise ToolNotFoundError(f"Tool {spec.name!r} is not available for scope {scope.value if scope else scope}.")
        return spec

    def _allowed_for_scope(self, spec: ToolSpec, scope: AgentScope | None) -> bool:
        return scope is None or not spec.scopes or scope in spec.scopes

    def _coerce_tool(
        self,
        obj: BaseTool | Callable[..., Any],
        *,
        name: str | None,
        description: str | None,
    ) -> BaseTool:
        if isinstance(obj, BaseTool):
            return self._clone_tool(obj, name or obj.name, description)

        kwargs = {
            "name": name or obj.__name__,
            "description": description or inspect.getdoc(obj) or f"{name or obj.__name__} tool",
        }
        if inspect.iscoroutinefunction(obj):
            return StructuredTool.from_function(coroutine=obj, **kwargs)
        return StructuredTool.from_function(func=obj, **kwargs)

    def _clone_tool(self, lc_tool: BaseTool, name: str, description: str | None = None) -> BaseTool:
        copied = lc_tool.model_copy(deep=True)
        copied.name = name
        if description:
            copied.description = description
        return copied


def _dedupe_specs(specs: list[ToolSpec]) -> list[ToolSpec]:
    seen: set[str] = set()
    result: list[ToolSpec] = []
    for spec in specs:
        key = f"{spec.name}:{spec.public_name}"
        if key in seen:
            continue
        seen.add(key)
        result.append(spec)
    return result
