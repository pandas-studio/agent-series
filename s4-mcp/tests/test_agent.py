import os
import sys
import warnings

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from docs_mcp import server
from docs_mcp.agent import ask, build_agent, load_settings, server_config

warnings.filterwarnings("ignore", message=".*langchain.mcp.*beta.*")
from langchain.mcp import MCPAdapter  # noqa: E402

LIVE = pytest.mark.skipif(os.environ.get("AGENT_LIVE") != "1", reason="needs the local model server (AGENT_LIVE=1)")


@pytest.fixture
def docs(tmp_path, monkeypatch):
    (tmp_path / "langgraph").mkdir()
    (tmp_path / "langgraph" / "persistence.mdx").write_text(
        "---\ntitle: Persistence\n---\nA checkpointer saves graph state.\nUse a thread_id to resume.\n")
    monkeypatch.setenv("DOCS_ROOT", str(tmp_path))
    return tmp_path


class ToolCallingFake(GenericFakeChatModel):
    """Plays back fixed AI messages; create_agent binds tools, which a fake model ignores."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages: AIMessage) -> ToolCallingFake:
    return ToolCallingFake(messages=iter(messages))


def call(name: str, args: dict, id_: str = "c1") -> dict:
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


def test_server_config_is_an_mcpconfig_without_transport_key(tmp_path):
    cfg = server_config()
    entry = cfg["mcpServers"]["docs"]
    assert entry == {"command": sys.executable, "args": ["-m", "docs_mcp.server"]}
    assert server_config(tmp_path)["mcpServers"]["docs"]["env"] == {"DOCS_ROOT": str(tmp_path)}


def test_settings_from_env():
    s = load_settings({"AGENT_BASE_URL": "http://x/v1", "AGENT_MODEL": "m", "AGENT_TEMPERATURE": "0"})
    assert (s.base_url, s.model, s.temperature) == ("http://x/v1", "m", 0.0)


async def test_list_tools_returns_async_structured_tools(docs):
    async with MCPAdapter(server.mcp) as adapter:
        tools = {t.name: t for t in await adapter.list_tools()}
    assert set(tools) == {"search_docs", "read_doc"}
    assert tools["read_doc"].args["path"] == {"type": "string"}
    assert tools["read_doc"].metadata["mcp"]["server"]["name"] == "docs"
    with pytest.raises(NotImplementedError):
        tools["search_docs"].invoke({"query": "checkpointer"})


async def test_tools_stay_callable_after_the_adapter_exits(docs):
    async with MCPAdapter(server.mcp) as adapter:
        tools = {t.name: t for t in await adapter.list_tools()}
    msg = await tools["search_docs"].ainvoke(call("search_docs", {"query": "checkpointer"}))
    assert msg.status == "success"
    assert msg.artifact["structured_content"]["result"][0]["path"] == "langgraph/persistence.mdx"


async def test_server_error_becomes_an_error_tool_message(docs):
    async with MCPAdapter(server.mcp) as adapter:
        tools = {t.name: t for t in await adapter.list_tools()}
    msg = await tools["read_doc"].ainvoke(call("read_doc", {"path": "../secret.txt"}))
    assert msg.status == "error" and "unknown page" in msg.text


async def test_agent_searches_reads_and_answers_through_mcp(docs):
    async with MCPAdapter(server.mcp) as adapter:
        tools = await adapter.list_tools()
    model = scripted(
        AIMessage("", tool_calls=[call("search_docs", {"query": "checkpointer"}, "s1")]),
        AIMessage("", tool_calls=[call("read_doc", {"path": "langgraph/persistence.mdx"}, "r1")]),
        AIMessage("checkpointer는 그래프 상태를 저장합니다 (langgraph/persistence.mdx)."),
    )
    out = await build_agent(tools, model).ainvoke({"messages": [{"role": "user", "content": "q"}]})
    results = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in results] == ["search_docs", "read_doc"]
    assert all(m.status == "success" for m in results)
    assert "A checkpointer saves graph state." in results[1].text
    assert out["messages"][-1].content.endswith("(langgraph/persistence.mdx).")


async def test_agent_reads_the_server_error_and_continues(docs):
    async with MCPAdapter(server.mcp) as adapter:
        tools = await adapter.list_tools()
    model = scripted(
        AIMessage("", tool_calls=[call("read_doc", {"path": "langgraph/memory.mdx"}, "r1")]),
        AIMessage("", tool_calls=[call("search_docs", {"query": "memory checkpointer"}, "s1")]),
        AIMessage("langgraph/persistence.mdx 를 보세요."),
    )
    out = await build_agent(tools, model).ainvoke({"messages": [{"role": "user", "content": "q"}]})
    results = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in results] == ["error", "success"]


def test_sync_agent_invoke_fails_with_mcp_tools(docs):
    import asyncio

    async def discover():
        async with MCPAdapter(server.mcp) as adapter:
            return await adapter.list_tools()

    model = scripted(AIMessage("", tool_calls=[call("search_docs", {"query": "checkpointer"})]), AIMessage("done"))
    agent = build_agent(asyncio.run(discover()), model)
    with pytest.raises(NotImplementedError, match="does not support sync invocation"):
        agent.invoke({"messages": [{"role": "user", "content": "q"}]})


async def test_stdio_server_needs_docs_root_in_its_env(docs):
    """The stdio subprocess does not see the parent's DOCS_ROOT; the MCPConfig env entry passes it."""
    (docs / "quokka.mdx").write_text("# Quokka\nquokka marks this test folder\n")

    async def found(target) -> list:
        async with MCPAdapter(target) as adapter:
            search = {t.name: t for t in await adapter.list_tools()}["search_docs"]
        msg = await search.ainvoke(call("search_docs", {"query": "quokka"}))
        return [h["path"] for h in msg.artifact["structured_content"]["result"]]

    assert await found(server_config(docs)) == ["quokka.mdx"]
    assert await found(server_config()) == []   # the server searched its default folder


@LIVE
async def test_live_agent_answers_from_the_docs():
    out = await ask("LangGraph에서 checkpointer는 무엇을 저장하나요?")
    names = [c["name"] for m in out["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
    assert "search_docs" in names
    assert out["messages"][-1].content
