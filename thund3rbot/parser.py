"""
ResponseParser — unified LLM response parser for all agent scopes.

Normalizes output from any provider into a structured ParseResult, working
around local-model inconsistency (text tool calls vs API function-calling).

Priority chain
--------------
1. native — response.tool_calls already populated (cloud providers via LangChain)
2. xml    — <tool_call>…</tool_call> blocks  (format taught by _TOOL_FORMAT_INSTRUCTIONS)
3. json   — JSON objects found in text  (single, concatenated, or prose-embedded)
4. plain  — no tool calls; entire text is the final answer
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── compiled patterns ─────────────────────────────────────────────────────────
_THINK_RE    = re.compile(r"<think>(.*?)</think>",        re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_TOOL_RESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL | re.IGNORECASE)
_FENCE_RE    = re.compile(r"```\w*\n?|```")


# ── public data types ─────────────────────────────────────────────────────────

@dataclass
class ToolCallSpec:
    """Canonical in-flight tool call extracted by the parser."""
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def as_langchain(self) -> dict[str, Any]:
        """Return the dict format that LangChain AIMessage / ToolMessage expect."""
        return {"id": self.id, "name": self.name, "args": self.args, "type": "tool_call"}


@dataclass
class ParseResult:
    """
    Structured output from ResponseParser.parse().

    Attributes
    ----------
    mode        : How the result was decoded.
                  One of "native" | "xml" | "json" | "plain".
    tool_calls  : Extracted tool calls (empty for plain-text responses).
    reasoning   : Content of <think>…</think> blocks, if present.
    response    : Visible text after tool-call markup is stripped.
    raw_content : Original content string, unchanged.
    diagnostics : Metadata useful for logging and debugging.
    """
    mode: str
    tool_calls: list[ToolCallSpec] = field(default_factory=list)
    reasoning: str = ""
    response: str = ""
    raw_content: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def langchain_tool_calls(self) -> list[dict[str, Any]]:
        """Return tool_calls in the format expected by LangChain AIMessage."""
        return [tc.as_langchain() for tc in self.tool_calls]


# ── parser ────────────────────────────────────────────────────────────────────

class ResponseParser:
    """
    Convert an LLM response (LangChain AIMessage or similar) into a ParseResult.

    Parameters
    ----------
    tool_names
        Names of tools available to this agent.  Used to validate extracted
        tool names and to apply the create_task_agent remap heuristic.
        Pass ``None`` to skip name validation entirely.
    """

    def __init__(self, tool_names: Optional[set[str]] = None) -> None:
        self._tool_names = tool_names  # None → accept any name

    # ── public API ────────────────────────────────────────────────────────────

    def parse(self, response: Any) -> ParseResult:
        """
        Parse *response* and return a ParseResult.

        Tries the priority chain: native → xml → json → plain.
        Never raises; always returns a valid ParseResult.
        """
        content = getattr(response, "content", "") or ""
        raw: str = content if isinstance(content, str) else ""

        # ── 1. Native tool calls (OpenAI, Anthropic, etc.) ───────────────────
        native = getattr(response, "tool_calls", None)
        if native:
            calls = [self._native_to_spec(tc, i) for i, tc in enumerate(native)]
            logger.debug("ResponseParser mode=native  calls=%d", len(calls))
            return ParseResult(
                mode="native",
                tool_calls=calls,
                raw_content=raw,
                diagnostics={"mode": "native"},
            )

        if not raw.strip():
            return ParseResult(mode="plain", raw_content=raw, diagnostics={"mode": "plain"})

        # Extract reasoning from <think> tags; they are not part of the response.
        reasoning, text = _extract_thinking(raw)

        # ── 2. <tool_call> XML blocks ─────────────────────────────────────────
        xml_calls, xml_clean = self._parse_xml(text)
        if xml_calls:
            logger.debug("ResponseParser mode=xml  calls=%d", len(xml_calls))
            return ParseResult(
                mode="xml",
                tool_calls=xml_calls,
                reasoning=reasoning,
                response=xml_clean,
                raw_content=raw,
                diagnostics={"mode": "xml", "count": len(xml_calls)},
            )

        # Strip <tool_response> wrappers and markdown fences before JSON parsing
        text = _strip_wrappers(text)

        # ── 3. JSON objects in text ───────────────────────────────────────────
        json_calls = self._parse_json(text)
        if json_calls:
            logger.debug("ResponseParser mode=json  calls=%d", len(json_calls))
            return ParseResult(
                mode="json",
                tool_calls=json_calls,
                reasoning=reasoning,
                raw_content=raw,
                diagnostics={"mode": "json", "count": len(json_calls)},
            )

        # ── 4. Plain text ─────────────────────────────────────────────────────
        return ParseResult(
            mode="plain",
            reasoning=reasoning,
            response=text.strip() or raw.strip(),
            raw_content=raw,
            diagnostics={"mode": "plain"},
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _native_to_spec(self, tc: Any, idx: int) -> ToolCallSpec:
        """Convert a LangChain tool_call dict/object to a ToolCallSpec."""
        if isinstance(tc, dict):
            name = tc.get("name", "")
            args = tc.get("args", {})
            cid  = tc.get("id", "")
        else:
            name = getattr(tc, "name", "")
            args = getattr(tc, "args", {})
            cid  = getattr(tc, "id", "")
        if not isinstance(args, dict):
            args = {}
        return ToolCallSpec(id=cid or _new_id(idx), name=str(name), args=args)

    def _parse_xml(self, text: str) -> tuple[list[ToolCallSpec], str]:
        """Extract <tool_call>…</tool_call> blocks; return (calls, cleaned_text)."""
        hits = list(_TOOL_CALL_RE.finditer(text))
        if not hits:
            return [], text

        calls: list[ToolCallSpec] = []
        for i, m in enumerate(hits):
            inner = _FENCE_RE.sub("", m.group(1)).strip()
            try:
                obj = json.loads(inner)
            except json.JSONDecodeError:
                continue
            spec = self._normalise(obj, i)
            if spec:
                calls.append(spec)

        clean = _TOOL_CALL_RE.sub("", text).strip()
        return calls, clean

    def _parse_json(self, text: str) -> list[ToolCallSpec]:
        """Extract all top-level JSON objects from *text* and normalise them."""
        objs = _extract_json_objects(text)
        calls: list[ToolCallSpec] = []
        for i, obj in enumerate(objs):
            spec = self._normalise(obj, i)
            if spec:
                calls.append(spec)
        return calls

    def _normalise(self, obj: dict[str, Any], idx: int) -> Optional[ToolCallSpec]:
        """
        Convert a raw parsed dict to a ToolCallSpec.

        Applies common field aliases (tool/function → name, parameters/input → args)
        and the create_task_agent remap heuristic for sub-agent delegation.
        Returns None if the object is not a valid tool call.
        """
        name = str(
            obj.get("name") or obj.get("tool") or obj.get("function") or ""
        ).strip()

        raw_args = (
            obj.get("arguments") or obj.get("args")
            or obj.get("parameters") or obj.get("input") or {}
        )
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {}
        args: dict = raw_args if isinstance(raw_args, dict) else {}

        if not name:
            return None

        if self._tool_names is not None and name not in self._tool_names:
            # Heuristic: model used a sub-agent/task name as the function name.
            # Remap to create_task_agent when the args carry a "task" key.
            if "task" in args and "create_task_agent" in self._tool_names:
                args.setdefault("name", name)
                name = "create_task_agent"
            else:
                return None  # unknown tool — discard

        cid = str(obj.get("call_id") or obj.get("id") or _new_id(idx))
        return ToolCallSpec(id=cid, name=name, args=args)


# ── module-level helpers ──────────────────────────────────────────────────────

def _new_id(idx: int = 0) -> str:
    return f"tc_{uuid.uuid4().hex[:8]}_{idx}"


def _extract_thinking(text: str) -> tuple[str, str]:
    """Return (thinking_text, text_with_think_tags_removed)."""
    parts = [m.group(1).strip() for m in _THINK_RE.finditer(text)]
    cleaned = _THINK_RE.sub("", text).strip()
    return "\n".join(parts), cleaned


def _strip_wrappers(text: str) -> str:
    """Strip <tool_response>…</tool_response> wrappers and markdown fences."""
    tr = _TOOL_RESP_RE.search(text)
    if tr:
        text = tr.group(1).strip()
    text = _FENCE_RE.sub("", text).strip()
    return text


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """
    Find and parse every top-level JSON object ``{…}`` in *text*.

    Handles concatenated objects (``{…}{…}``) and objects embedded in prose.
    """
    results: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < len(text):
            ch = text[j]
            if esc:
                esc = False
            elif ch == "\\" and in_str:
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[i : j + 1])
                            if isinstance(obj, dict):
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            break  # unterminated JSON object — stop
    return results
