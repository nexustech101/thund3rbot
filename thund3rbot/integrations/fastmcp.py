"""FastMCP adapter helpers."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from thund3rbot import AgentFactory, AgentScope


def register_fastmcp_tools(factory: AgentFactory, mcp: Any, names: Iterable[str] | None = None) -> None:
    """Expose Factory tools on a supplied FastMCP server instance."""

    selected = set(names) if names is not None else None
    for spec in factory.tools.list():
        if selected is not None and spec.name not in selected:
            continue
        source = spec.source

        if source is None:
            source = _wrap_langchain_tool(spec.tool, spec.name, spec.description)

        mcp.tool()(source)


async def load_mcp_tools(
    factory: AgentFactory,
    url: str,
    *,
    namespace: str | None = None,
    names: Iterable[str] | None = None,
    scopes: Iterable[AgentScope | str] | None = None,
    overrides: dict[str, str] | None = None,
) -> list[Any]:
    """Load tools from an MCP HTTP server into the factory registry."""

    from langchain_mcp_adapters.tools import load_mcp_tools as load_langchain_mcp_tools
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    selected = set(names) if names is not None else None
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_langchain_mcp_tools(session)

    if selected is not None:
        tools = [tool for tool in tools if tool.name in selected]
    for mcp_tool in tools:
        registered_name = f"{namespace}.{mcp_tool.name}" if namespace else mcp_tool.name
        factory.tools.add(
            mcp_tool,
            name=registered_name,
            public_name=mcp_tool.name,
            description=(overrides or {}).get(registered_name, getattr(mcp_tool, "description", "")),
            scopes=scopes,
            namespace=namespace,
        )
    return tools


async def load_mcp_prompts(
    factory: AgentFactory,
    url: str,
    *,
    namespace: str,
    names: Iterable[str] | None = None,
) -> list[Any]:
    """Load MCP prompt metadata into the factory prompt registry."""

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    selected = set(names) if names is not None else None
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_prompts()

    prompts = list(getattr(response, "prompts", response) or [])
    loaded = []
    for item in prompts:
        name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None)
        if not name or (selected is not None and name not in selected):
            continue
        description = getattr(item, "description", None) or (item.get("description") if isinstance(item, dict) else "")
        full_name = f"{namespace}.{name}"
        factory.prompts.register(
            f"MCP prompt {full_name}. Fetch prompt content from the MCP server before use.",
            name=name,
            description=description or "",
            namespace=namespace,
        )
        loaded.append(factory.prompts.get(full_name))
    return [item for item in loaded if item is not None]


def _wrap_langchain_tool(tool: Any, name: str, description: str):
    async def invoke_tool(**kwargs: Any) -> Any:
        return await tool.ainvoke(kwargs)

    invoke_tool.__name__ = name
    invoke_tool.__doc__ = description
    return invoke_tool
