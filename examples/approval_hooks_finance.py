from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage

from thund3rbot import (
    AgentFramework,
    AgentScope,
    AgentSpec,
    FrameworkConfig,
    ModelConfig,
    RunOptions,
    ToolApproval,
    tool,
)


class ScriptedModel:
    def __init__(self):
        self.responses = [
            '<tool_call>{"name": "schedule_transfer", "arguments": '
            '{"vendor": "Northwind Hosting", "amount_usd": 1250.00}}</tool_call>',
            "I did not schedule the transfer because approval is required.",
        ]

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content=self.responses.pop(0))


@tool(scopes=[AgentScope.TASK], risk="high", requires_approval=True, tags=["finance"])
def schedule_transfer(vendor: str, amount_usd: float) -> str:
    """Schedule a vendor payment."""

    return f"Scheduled ${amount_usd:,.2f} transfer to {vendor}."


def require_manual_review(context):
    amount = context.arguments.get("amount_usd", 0)
    if context.risk == "high" or amount >= 500:
        return ToolApproval(
            approved=False,
            reason="finance transfers over $500 require manual approval",
        )
    return None


async def main() -> None:
    framework = AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider="custom", model="scripted"),
            model_factory=lambda _: ScriptedModel(),
        )
    )
    framework.tools.register(schedule_transfer)

    agent = framework.agent(
        AgentSpec(
            name="finance_assistant",
            scope=AgentScope.TASK,
            instructions="Help prepare finance operations. Never hide rejected actions.",
            tools=["schedule_transfer"],
        )
    )

    result = await agent.run(
        "Pay Northwind Hosting for this month's invoice.",
        options=RunOptions(before_tool_call=require_manual_review),
    )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
