import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from docs_mcp import server

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def docs(tmp_path, monkeypatch):
    (tmp_path / "langgraph").mkdir()
    (tmp_path / "langgraph" / "persistence.mdx").write_text(
        "---\ntitle: Persistence\n---\nA checkpointer saves graph state.\nUse a thread_id to resume.\n")
    (tmp_path / "langgraph" / "streaming.mdx").write_text("# Streaming\nStream graph state as it runs.\n" + "x" * 500)
    (tmp_path / "secret.txt").write_text("not a doc page")
    monkeypatch.setenv("DOCS_ROOT", str(tmp_path))
    return tmp_path


async def test_lists_two_tools(docs):
    async with Client(server.mcp) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == {"search_docs", "read_doc"}


async def test_tool_schema_comes_from_type_hints(docs):
    async with Client(server.mcp) as c:
        tool = {t.name: t for t in await c.list_tools()}["read_doc"]
    props = tool.input_schema["properties"]
    assert props["path"]["type"] == "string" and props["start"]["type"] == "integer"
    assert tool.input_schema["required"] == ["path"]
    assert "search_docs" in tool.description


async def test_search_ranks_and_reads(docs):
    async with Client(server.mcp) as c:
        hits = (await c.call_tool("search_docs", {"query": "checkpointer thread_id"})).data
        assert hits[0]["path"] == "langgraph/persistence.mdx"
        assert hits[0]["title"] == "Persistence"
        page = (await c.call_tool("read_doc", {"path": hits[0]["path"]})).data
    assert "thread_id" in page["text"] and page["more"] is False


async def test_read_in_parts(docs):
    async with Client(server.mcp) as c:
        first = (await c.call_tool("read_doc", {"path": "langgraph/streaming.mdx", "max_chars": 200})).data
        second = (await c.call_tool("read_doc", {"path": "langgraph/streaming.mdx", "start": first["next_start"],
                                                 "max_chars": 200})).data
    assert first["more"] and first["next_start"] == 200 and second["start"] == 200


@pytest.mark.parametrize("path", ["../../etc/passwd", "secret.txt", "langgraph/missing.mdx"])
async def test_refuses_paths_outside_docs(docs, path):
    async with Client(server.mcp) as c:
        result = await c.call_tool("read_doc", {"path": path}, raise_on_error=False)
    assert result.is_error
    assert "unknown page" in result.content[0].text


async def test_over_stdio_subprocess(docs):
    # The same server as a separate process, the way Claude Code or an agent starts it.
    transport = StdioTransport(command=sys.executable, args=["-m", "docs_mcp.server"],
                               env={"DOCS_ROOT": str(docs)}, cwd=str(PROJECT))
    async with Client(transport) as c:
        hits = (await c.call_tool("search_docs", {"query": "stream"})).data
    assert hits[0]["path"] == "langgraph/streaming.mdx"
