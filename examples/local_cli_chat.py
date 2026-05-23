from __future__ import annotations

import asyncio
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thund3rbot import AgentFramework, AgentScope, AgentSpec, FrameworkConfig, ModelConfig


@lru_cache(maxsize=1)
def create_assistant(
    *,
    provider: str = "ollama",
    model: str = "qwen3.5:9b",
):
    framework = AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider=provider, model=model),
        )
    )
    return framework.agent(
        AgentSpec(
            name="assistant",
            scope=AgentScope.TASK,
            session_id="local-cli",
            instructions="Answer questions concisely using clear language.",
        )
    )


async def main() -> None:
    agent = create_assistant()
    print("Thund3rBot CLI. Type 'exit' to quit.")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = await agent.run(user_input)
        if result.error:
            print(f"error: {result.error}")
            continue
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
