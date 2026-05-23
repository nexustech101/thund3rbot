"""Skill registration, composition, and Markdown/YAML discovery."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from thund3rbot.types import AgentScope, Skill, SkillConfigError, ToolNotFoundError


class SkillRegistry:
    """Runtime-local skill registry."""

    def __init__(self, framework: Any | None = None) -> None:
        self._framework = framework
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill | str, **kwargs: Any) -> Skill:
        value = skill if isinstance(skill, Skill) else Skill(name=skill, **kwargs)
        if value.scope and not value.scopes:
            value.scopes = {value.scope}
        self._skills[value.name] = value
        try:
            self._validate_no_cycles()
        except Exception:
            self._skills.pop(value.name, None)
            raise
        return value

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def require(self, name: str) -> Skill:
        skill = self.get(name)
        if skill is None:
            available = ", ".join(sorted(self._skills)) or "(none)"
            raise KeyError(f"Skill {name!r} is not registered. Available: {available}")
        return skill

    def resolve(self, names: list[str], *, scope: AgentScope | None = None) -> list[Skill]:
        resolved: list[Skill] = []
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise SkillConfigError(f"Circular skill dependency involving {name!r}.")
            skill = self.require(name)
            visiting.add(name)
            for dependency in skill.requires:
                visit(dependency)
            visiting.remove(name)
            if scope is not None and skill.scopes and scope not in skill.scopes:
                raise SkillConfigError(f"Skill {name!r} is not available for scope {scope.value!r}.")
            if skill.name not in {item.name for item in resolved}:
                resolved.append(skill)

        for name in names:
            visit(name)
        return resolved

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def load_dir(self, path: str | Path) -> list[Skill]:
        root = Path(path)
        if not root.exists():
            return []
        loaded: list[Skill] = []
        for file in sorted(root.glob("*.md")):
            skill = parse_skill_file(file)
            self._validate_markdown_tools(skill)
            self.register(skill)
            loaded.append(skill)
        return loaded

    def _validate_markdown_tools(self, skill: Skill) -> None:
        if self._framework is None:
            return
        for ref in skill.tools:
            if isinstance(ref, str) and not ref.endswith(".*") and ref not in self._framework.tools:
                raise ToolNotFoundError(f"Skill {skill.name!r} references unknown tool {ref!r}.")

    def _validate_no_cycles(self) -> None:
        visited: set[str] = set()
        stack: set[str] = set()

        def visit(name: str) -> None:
            if name in stack:
                raise SkillConfigError(f"Circular skill dependency involving {name!r}.")
            if name in visited:
                return
            stack.add(name)
            skill = self._skills.get(name)
            if skill:
                for dependency in skill.requires:
                    if dependency in self._skills:
                        visit(dependency)
            stack.remove(name)
            visited.add(name)

        for skill_name in list(self._skills):
            visit(skill_name)


def parse_skill_file(path: str | Path) -> Skill:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {
        "name": file.stem,
        "description": "",
        "tools": [],
        "requires": [],
        "scopes": [],
    }
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            metadata.update(_parse_frontmatter(parts[1]))
            body = parts[2].strip()

    scope_raw = metadata.get("scope")
    scopes_raw = metadata.get("scopes") or ([scope_raw] if scope_raw else [])
    scopes = {AgentScope(scope) for scope in scopes_raw}
    return Skill(
        name=str(metadata.get("name") or file.stem),
        description=str(metadata.get("description") or ""),
        instructions=body,
        tools=list(metadata.get("tools") or []),
        requires=list(metadata.get("requires") or []),
        scopes=scopes,
        scope=next(iter(scopes), None),
        metadata={
            "path": str(file),
            **{
                k: v
                for k, v in metadata.items()
                if k not in {"name", "description", "tools", "requires", "scope", "scopes"}
            },
        },
    )


def _parse_frontmatter(text: str) -> dict[str, Any]:
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        result: dict[str, Any] = {}
        current_list_key: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- ") and current_list_key:
                result.setdefault(current_list_key, []).append(stripped[2:].strip().strip("\"'"))
                continue
            if ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip("\"'")
            if value.startswith("[") and value.endswith("]"):
                result[key] = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
                current_list_key = None
            elif value:
                result[key] = value
                current_list_key = None
            else:
                result[key] = []
                current_list_key = key
        return result
