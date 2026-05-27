from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from thund3rbot import AgentFactory, AgentScope, AgentSpec, FactoryConfig, ModelConfig


class SentimentResult(BaseModel):
    sentiment: str
    urgency: int
    route_to: str
    summary: str


class ScriptedModel:
    async def ainvoke(self, messages):
        return AIMessage(
            content=(
                '{"sentiment": "frustrated", "urgency": 4, '
                '"route_to": "customer_success", '
                '"summary": "Customer is blocked and needs a fast follow-up."}'
            )
        )


async def main() -> None:
    framework = AgentFactory(
        FactoryConfig(
            default_model=ModelConfig(provider="custom", model="scripted"),
            model_factory=lambda _: ScriptedModel(),
        )
    )
    agent = framework.agent(
        AgentSpec(
            name="support_triage",
            scope=AgentScope.TASK,
            instructions="Classify customer sentiment and return the requested schema.",
            output_schema=SentimentResult,
        )
    )

    result = await agent.run(
        "This is the third time I have asked for help and our launch is tomorrow."
    )
    print(result.output.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
