"""Prompt registry and public ``@prompt`` decorator."""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from thund3rbot.types import PromptSpec


@dataclass(frozen=True)
class PromptMetadata:
    name: str | None = None
    description: str | None = None


def prompt(
    func: Callable[..., str] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
):
    """Decorate a Python function as a framework prompt template."""

    def _decorate(fn: Callable[..., str]) -> Callable[..., str]:
        setattr(fn, "__agent_prompt_metadata__", PromptMetadata(name, description))
        return fn

    if func is None:
        return _decorate
    return _decorate(func)


class PromptRegistry:
    """Runtime-local prompt registry."""

    def __init__(self, framework: Any | None = None) -> None:
        self._framework = framework
        self._prompts: dict[str, PromptSpec] = {}

    def register(
        self,
        prompt_or_func: Callable[..., str] | str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        namespace: str | None = None,
    ):
        """Register a prompt directly or as a decorator."""

        def _register(obj: Callable[..., str] | str):
            metadata: PromptMetadata | None = getattr(obj, "__agent_prompt_metadata__", None)
            prompt_name = name or (metadata.name if metadata else None)
            if not prompt_name:
                prompt_name = obj.__name__ if callable(obj) else "prompt"
            full_name = f"{namespace}.{prompt_name}" if namespace else prompt_name
            prompt_description = description or (metadata.description if metadata else None)
            if not prompt_description and callable(obj):
                prompt_description = inspect.getdoc(obj) or ""
            self._prompts[full_name] = PromptSpec(
                name=full_name,
                description=prompt_description or "",
                prompt=obj,
                namespace=namespace,
            )
            return obj

        if prompt_or_func is None:
            return _register
        return _register(prompt_or_func)

    def get(self, name: str) -> PromptSpec | None:
        return self._prompts.get(name)

    def list(self) -> list[PromptSpec]:
        return list(self._prompts.values())

    def names(self) -> list[str]:
        return sorted(self._prompts)

    def render(self, name: str, **kwargs: Any) -> str:
        spec = self._prompts[name]
        if callable(spec.prompt):
            return spec.prompt(**kwargs)
        return spec.prompt.format(**kwargs) if kwargs else spec.prompt

    async def load_mcp(
        self,
        url: str,
        *,
        namespace: str,
        names: Iterable[str] | None = None,
    ) -> list[PromptSpec]:
        """Load MCP prompts under a namespace without importing MCP until called."""

        if self._framework is None:
            raise RuntimeError("MCP loading requires a PromptRegistry attached to an AgentFramework.")
        from thund3rbot.integrations.fastmcp import load_mcp_prompts

        return await load_mcp_prompts(self._framework, url, namespace=namespace, names=names)
