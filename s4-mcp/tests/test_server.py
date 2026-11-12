import json
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mcp.shared.exceptions import MCPError

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


async def test_lists_resources_and_prompts(docs):
    async with Client(server.mcp) as c:
        resources = [str(r.uri) for r in await c.list_resources()]
        templates = [t.uri_template for t in await c.list_resource_templates()]
        prompts = {p.name: [a.name for a in p.arguments] for p in await c.list_prompts()}
    assert resources == ["docs://index"]
    assert templates == ["docs://page/{path*}"]
    assert prompts == {"ask_docs": ["question"], "summarize_page": ["path"]}


async def test_index_lists_every_page(docs):
    async with Client(server.mcp) as c:
        pages = json.loads((await c.read_resource("docs://index"))[0].text)
    assert [p["path"] for p in pages] == ["langgraph/persistence.mdx", "langgraph/streaming.mdx"]
    assert pages[0]["uri"] == "docs://page/langgraph/persistence.mdx" and pages[0]["title"] == "Persistence"


async def test_page_resource_reads_a_nested_path(docs):
    # {path*} spans the "/" in langgraph/persistence.mdx; a plain {path} would not match this URI.
    async with Client(server.mcp) as c:
        content = (await c.read_resource("docs://page/langgraph/persistence.mdx"))[0]
    assert content.mime_type == "text/markdown"
    assert content.text == (docs / "langgraph" / "persistence.mdx").read_text()


@pytest.mark.parametrize("path", ["secret.txt", "langgraph/missing.mdx", "../../etc/passwd"])
async def test_page_resource_refuses_paths_outside_docs(docs, path):
    async with Client(server.mcp) as c:
        with pytest.raises(MCPError, match="unknown page"):
            await c.read_resource(f"docs://page/{path}")


async def test_summarize_page_embeds_the_page(docs):
    async with Client(server.mcp) as c:
        result = await c.get_prompt("summarize_page", {"path": "langgraph/persistence.mdx"})
    first, second = result.messages
    assert first.content.type == "resource"
    assert str(first.content.resource.uri) == "docs://page/langgraph/persistence.mdx"
    assert "thread_id" in first.content.resource.text
    assert second.content.type == "text" and "Korean" in second.content.text


async def test_ask_docs_names_the_tools(docs):
    async with Client(server.mcp) as c:
        result = await c.get_prompt("ask_docs", {"question": "What does a checkpointer save?"})
    text = result.messages[0].content.text
    assert "What does a checkpointer save?" in text
    assert "search_docs" in text and "read_doc" in text


async def test_over_stdio_subprocess(docs):
    # The same server as a separate process, the way Claude Code or an agent starts it.
    transport = StdioTransport(command=sys.executable, args=["-m", "docs_mcp.server"],
                               env={"DOCS_ROOT": str(docs)}, cwd=str(PROJECT))
    async with Client(transport) as c:
        hits = (await c.call_tool("search_docs", {"query": "stream"})).data
        page = (await c.read_resource("docs://page/langgraph/streaming.mdx"))[0].text
    assert hits[0]["path"] == "langgraph/streaming.mdx"
    assert page.startswith("# Streaming")


async def test_project_mcp_json_starts_the_server():
    # The repository's .mcp.json, started the way Claude Code starts a project-scope server: from the repository root.
    repo = PROJECT.parent
    entry = json.loads((repo / ".mcp.json").read_text())["mcpServers"]["docs"]
    assert entry["type"] == "stdio" and "cwd" not in entry   # Claude Code ignores "cwd"; --directory picks the project
    transport = StdioTransport(command=entry["command"], args=entry["args"], cwd=str(repo))
    async with Client(transport) as c:
        assert {t.name for t in await c.list_tools()} == {"search_docs", "read_doc"}
