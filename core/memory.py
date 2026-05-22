"""
Memory and context management for agents.

Architecture:
  BaseMemoryStore — abstract interface
  InMemoryStore   — default dev implementation (no persistence)

Swap out InMemoryStore for a Redis / Postgres backed store at any time
by calling set_memory_store() before starting your application.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────


class BaseMemoryStore(ABC):
    """
    Interface for agent memory.

    session_id is typically the agent_id, giving each agent its own memory lane.
    """

    @abstractmethod
    async def get_history(self, session_id: str) -> list[BaseMessage]:
        """Retrieve the full message history for a session."""
        ...

    @abstractmethod
    async def add_message(self, session_id: str, message: BaseMessage) -> None:
        """Append a single message to the session history."""
        ...

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Delete all history and context for a session."""
        ...

    @abstractmethod
    async def get_context(self, session_id: str) -> dict[str, Any]:
        """Retrieve arbitrary key-value context for a session."""
        ...

    @abstractmethod
    async def set_context(self, session_id: str, key: str, value: Any) -> None:
        """Store a single key-value pair in the session context."""
        ...

    async def get_context_value(self, session_id: str, key: str) -> Any:
        """Convenience: get a single context value."""
        return (await self.get_context(session_id)).get(key)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory implementation
# ─────────────────────────────────────────────────────────────────────────────


class InMemoryStore(BaseMemoryStore):
    """
    Simple in-process memory. Suitable for development and single-process deployments.

    For multi-process or persistent memory, replace with a Redis or DB-backed store.
    """

    def __init__(self) -> None:
        self._histories: dict[str, list[BaseMessage]] = {}
        self._contexts: dict[str, dict[str, Any]] = {}

    async def get_history(self, session_id: str) -> list[BaseMessage]:
        return list(self._histories.get(session_id, []))

    async def add_message(self, session_id: str, message: BaseMessage) -> None:
        self._histories.setdefault(session_id, []).append(message)

    async def clear(self, session_id: str) -> None:
        self._histories.pop(session_id, None)
        self._contexts.pop(session_id, None)

    async def get_context(self, session_id: str) -> dict[str, Any]:
        return dict(self._contexts.get(session_id, {}))

    async def set_context(self, session_id: str, key: str, value: Any) -> None:
        self._contexts.setdefault(session_id, {})[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor — swap backend at startup if needed
# ─────────────────────────────────────────────────────────────────────────────

_store: BaseMemoryStore = InMemoryStore()


def get_memory_store() -> BaseMemoryStore:
    """Return the active global memory store."""
    return _store


def set_memory_store(store: BaseMemoryStore) -> None:
    """
    Replace the global memory store.

    Call this *before* creating any agents, e.g. in your app startup handler:

        from core.memory import set_memory_store
        from my_stores import RedisMemoryStore
        set_memory_store(RedisMemoryStore(url="redis://localhost"))
    """
    global _store
    _store = store