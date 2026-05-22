"""Application-wide configuration loaded from environment / .env file."""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Provider keys ──────────────────────────────────────────────
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # ── Ollama ─────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # ── Default model ──────────────────────────────────────────────
    default_provider: str = "ollama"
    default_model: str = "qwen2.5-coder:7b"
    default_temperature: float = 0.7

    # ── FastAPI ────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_title: str = "Agent Framework API"
    api_version: str = "0.1.0"

    # ── MCP Server ─────────────────────────────────────────────────
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001

    # ── Logging ────────────────────────────────────────────────────
    log_level: str = "INFO"

    @property
    def mcp_url(self) -> str:
        return f"http://{self.mcp_host}:{self.mcp_port}"


# Singleton — import this everywhere
settings = Settings()