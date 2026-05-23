from __future__ import annotations

import asyncio
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage

from thund3rbot import AgentFramework, AgentScope, AgentSpec, Artifact, FrameworkConfig, ModelConfig, tool


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        self._skip = tag in {"script", "style", "noscript"}

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)

    def markdown(self) -> str:
        return "\n\n".join(self.parts)


class ScriptedModel:
    def __init__(self):
        self.responses = [
            '<tool_call>{"name": "page_to_markdown", "arguments": '
            '{"url": "https://example.com"}}</tool_call>',
            "The page was extracted into a markdown artifact.",
        ]

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content=self.responses.pop(0))


@tool(scopes=[AgentScope.TASK], tags=["web", "extraction"])
def page_to_markdown(url: str) -> Artifact:
    """Fetch a web page and return visible text as a markdown artifact."""

    request = Request(url, headers={"User-Agent": "thund3rbot-example/1.0"})
    with urlopen(request, timeout=10) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = TextExtractor()
    parser.feed(html)
    return Artifact(type="markdown", uri=url, data=parser.markdown())


async def main() -> None:
    framework = AgentFramework(
        FrameworkConfig(
            default_model=ModelConfig(provider="custom", model="scripted"),
            model_factory=lambda _: ScriptedModel(),
        )
    )
    framework.tools.register(page_to_markdown)

    agent = framework.agent(
        AgentSpec(
            name="web_extractor",
            scope=AgentScope.TASK,
            instructions="Extract web pages and summarize what was produced.",
            tools=["page_to_markdown"],
        )
    )

    result = await agent.run("Extract https://example.com to markdown.")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
