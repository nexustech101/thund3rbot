"""Framework-owned chat model factory."""
from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

from core.types import ModelConfig, ModelProvider, ProviderConfig

logger = logging.getLogger(__name__)


def create_llm(
    config: Optional[ModelConfig] = None,
    providers: dict[str, ProviderConfig] | None = None,
) -> BaseChatModel:
    """Instantiate a LangChain chat model from explicit framework config."""

    config = config or ModelConfig()
    provider_name = config.provider.value if isinstance(config.provider, ModelProvider) else str(config.provider)
    provider_cfg = (providers or {}).get(provider_name)

    base_kwargs: dict = {
        **(provider_cfg.extra_kwargs if provider_cfg else {}),
        "temperature": config.temperature,
        **config.extra_kwargs,
    }
    if config.max_tokens is not None:
        base_kwargs["max_tokens"] = config.max_tokens

    logger.debug("Creating LLM: provider=%s model=%s", provider_name, config.model)

    if provider_name == ModelProvider.OLLAMA.value:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model,
            base_url=_base_url(provider_cfg, "OLLAMA_BASE_URL", "http://localhost:11434"),
            **base_kwargs,
        )

    if provider_name == ModelProvider.OPENAI.value:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model,
            api_key=_api_key(provider_cfg, "OPENAI_API_KEY"),  # type: ignore[arg-type]
            **base_kwargs,
        )

    if provider_name == ModelProvider.GOOGLE.value:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=_api_key(provider_cfg, "GOOGLE_API_KEY"),  # type: ignore[arg-type]
            **base_kwargs,
        )

    if provider_name == ModelProvider.ANTHROPIC.value:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model,
            api_key=_api_key(provider_cfg, "ANTHROPIC_API_KEY"),  # type: ignore[arg-type]
            **base_kwargs,
        )

    raise ValueError(f"Unsupported model provider: {provider_name!r}")


def _api_key(provider: ProviderConfig | None, default_env: str) -> str:
    env_name = provider.api_key_env if provider and provider.api_key_env else default_env
    value = provider.api_key if provider and provider.api_key else os.getenv(env_name)
    if not value:
        raise EnvironmentError(f"{env_name} is not set. Add it to your environment or ProviderConfig.")
    return value


def _base_url(provider: ProviderConfig | None, default_env: str, fallback: str) -> str:
    return (provider.base_url if provider and provider.base_url else os.getenv(default_env)) or fallback

