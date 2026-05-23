from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thund3rbot import AgentFramework, AgentScope, AgentSpec, FrameworkConfig, ModelConfig
from thund3rbot.integrations.fastapi import create_agent_router

try:
    from fastapi import FastAPI
except ImportError as exc:
    raise SystemExit('Install FastAPI support with: pip install "thund3rbot[fastapi]"') from exc


framework = AgentFramework(
    FrameworkConfig(
        default_model=ModelConfig(provider="ollama", model="llama3.2"),
    )
)

framework.agent(
    AgentSpec(
        name="route_assistant",
        scope=AgentScope.TASK,
        instructions="Return concise, consistently formatted responses.",
    )
)

app = FastAPI(title="Thund3rBot Example API")
app.include_router(create_agent_router(framework), prefix="/api/v1")


# Run with:
# uvicorn examples.fastapi_agent_route:app --reload
