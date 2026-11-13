"""A LangChain agent that uses the docs server's tools through langchain.mcp.MCPAdapter.

create_agent builds a LangGraph graph (model node + tool node). The tools are not Python functions in this
process: MCPAdapter starts the docs server over stdio, lists its tools and wraps each one as an async LangChain tool
that sends tools/call to the server.

    uv run python -m docs_mcp.agent "LangGraph에서 checkpointer는 무엇을 저장하나요?"
"""

import asyncio
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# langchain.mcp is beta in langchain 1.5 and warns once per process on import.
warnings.filterwarnings("ignore", message=".*langchain.mcp.*beta.*")
from langchain.mcp import MCPAdapter  # noqa: E402

SYSTEM = (
    "You answer questions about LangGraph and LangChain from the local documentation. "
    "Search the docs with search_docs, read the best page with read_doc, then answer in Korean "
    "and name the page path you used."
)


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://localhost:8000/v1"
    model: str = "nvidia/Qwen3.6-35B-A3B-NVFP4"
    api_key: str = "local"  # vLLM accepts any key unless started with --api-key
    temperature: float = 0.6
    timeout: float = 120.0


def load_settings(env: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    base = Settings()
    return Settings(
        base_url=env.get("AGENT_BASE_URL", base.base_url),
        model=env.get("AGENT_MODEL", base.model),
        api_key=env.get("AGENT_API_KEY", base.api_key),
        temperature=float(env.get("AGENT_TEMPERATURE", base.temperature)),
        timeout=float(env.get("AGENT_TIMEOUT", base.timeout)),
    )


def make_model(settings: Settings | None = None) -> ChatOpenAI:
    s = settings or load_settings()
    return ChatOpenAI(model=s.model, base_url=s.base_url, api_key=s.api_key, temperature=s.temperature,
                      timeout=s.timeout,
                      # Qwen3 thinks by default; agent steps are short, so thinking is off.
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def server_config(docs_root: Path | None = None) -> dict:
    """MCPConfig for the docs server over stdio, started with this interpreter.

    The transport is inferred from the entry (command → stdio, url → HTTP); there is no transport key.
    """
    entry: dict = {"command": sys.executable, "args": ["-m", "docs_mcp.server"]}
    if docs_root is not None:
        entry["env"] = {"DOCS_ROOT": str(docs_root)}
    return {"mcpServers": {"docs": entry}}


def build_agent(tools, model=None):
    return create_agent(model or make_model(), tools, system_prompt=SYSTEM)


async def ask(question: str, target=None, model=None) -> dict:
    """Discover the tools, build the agent and answer one question.

    The agent runs after the adapter's `async with` has exited: the tools hold the client and reconnect on each
    call (with the default stdio keep_alive the server process stays up between calls).
    """
    async with MCPAdapter(target if target is not None else server_config()) as adapter:
        tools = await adapter.list_tools()
    agent = build_agent(tools, model)
    return await agent.ainvoke({"messages": [{"role": "user", "content": question}]})


def main(argv: list[str]) -> int:
    question = " ".join(argv) or "LangGraph에서 checkpointer는 무엇을 저장하나요?"
    out = asyncio.run(ask(question))
    for m in out["messages"]:
        for call in getattr(m, "tool_calls", None) or []:
            print(f"tool call: {call['name']} {call['args']}", file=sys.stderr)
    print(out["messages"][-1].content)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
