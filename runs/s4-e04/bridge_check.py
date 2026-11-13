"""Facts about the MCPAdapter bridge, checked without a model. Prints one JSON line per check.

Run from s4-mcp: uv run python ../runs/s4-e04/bridge_check.py > ../runs/s4-e04/bridge-check.jsonl.txt
"""

import asyncio
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

from langchain_core.messages import AIMessage

warnings.filterwarnings("ignore", message=".*langchain.mcp.*beta.*")
from langchain.mcp import MCPAdapter  # noqa: E402

from docs_mcp import server  # noqa: E402
from docs_mcp.agent import server_config  # noqa: E402


def out(check: str, **fields) -> None:
    print(json.dumps({"check": check} | fields, ensure_ascii=False, default=str), flush=True)


async def main() -> None:
    async with MCPAdapter(server_config()) as adapter:
        tools = await adapter.list_tools()
        client = adapter.client
        resources = [r.uri for r in await client.list_resources()]
        templates = [t.uri_template for t in await client.list_resource_templates()]
        prompts = [p.name for p in await client.list_prompts()]
    by = {t.name: t for t in tools}
    out("what list_tools returns", tools=sorted(by), resources_on_server=resources,
        templates_on_server=templates, prompts_on_server=prompts,
        adapter_methods=sorted(n for n in dir(MCPAdapter) if not n.startswith("_")))
    t = by["read_doc"]
    out("one adapted tool", type=type(t).__name__, description=t.description,
        args=t.args, response_format=t.response_format, metadata=t.metadata)

    # Two entries in one MCPConfig: FastMCP prefixes every tool with its config key.
    two = {"mcpServers": {"docs": server_config()["mcpServers"]["docs"],
                          "notes": server_config()["mcpServers"]["docs"]}}
    async with MCPAdapter(two) as adapter2:
        out("two servers in one MCPConfig", tools=sorted(x.name for x in await adapter2.list_tools()))

    # A server error (ToolError → isError=True) reaches the agent as a ToolMessage with status="error".
    # A tool given a ToolCall returns a ToolMessage, as ToolNode/create_agent call it.
    call = {"name": "read_doc", "args": {"path": "langgraph/no-such-page.mdx"}, "id": "c1", "type": "tool_call"}
    msg = await by["read_doc"].ainvoke(call)
    out("unknown page as a tool call", status=msg.status, content=msg.text, artifact=msg.artifact)

    # Successful call: model-visible content and the structured artifact.
    call_ok = {"name": "search_docs", "args": {"query": "checkpointer", "limit": 2}, "id": "c2", "type": "tool_call"}
    msg = await by["search_docs"].ainvoke(call_ok)
    out("search_docs as a tool call", status=msg.status, content_head=msg.text[:200],
        artifact_keys=sorted((msg.artifact or {}).keys()))

    # The adapted tools are async only.
    try:
        by["search_docs"].invoke({"query": "checkpointer"})
        out("sync tool.invoke", ok=True)
    except Exception as e:  # noqa: BLE001
        out("sync tool.invoke", ok=False, error=f"{type(e).__name__}: {e}")

    # A str target must be an http(s) URL.
    try:
        MCPAdapter("src/docs_mcp/server.py")
        out("str path target", ok=True)
    except Exception as e:  # noqa: BLE001
        out("str path target", ok=False, error=f"{type(e).__name__}: {e}")

    # The in-process server (tests): same tools, no subprocess.
    async with MCPAdapter(server.mcp) as mem:
        out("in-process FastMCP target", tools=sorted(x.name for x in await mem.list_tools()))

    # Does the stdio server see this process's DOCS_ROOT? Point it at a folder with one page and search for it.
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "zebra.mdx").write_text("---\ntitle: Zebra\n---\nzebra page\n")
        os.environ["DOCS_ROOT"] = tmp
        found = {}
        for label, target in (("no env entry", server_config()), ("env entry", server_config(Path(tmp)))):
            async with MCPAdapter(target) as a:
                search = {x.name: x for x in await a.list_tools()}["search_docs"]
            msg = await search.ainvoke({"name": "search_docs", "args": {"query": "zebra"}, "id": "z", "type": "tool_call"})
            found[label] = msg.artifact["structured_content"] if msg.artifact else msg.text
        del os.environ["DOCS_ROOT"]
        out("DOCS_ROOT set in the parent process only", results=found)


def sync_agent_invoke() -> None:
    """agent.invoke (sync) with MCP tools, as a sync LangGraph app would call it."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    from docs_mcp.agent import build_agent

    async def discover():
        async with MCPAdapter(server_config()) as adapter:
            return await adapter.list_tools()

    tools = asyncio.run(discover())

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    call = {"name": "search_docs", "args": {"query": "checkpointer"}, "id": "c3", "type": "tool_call"}
    model = Fake(messages=iter([AIMessage("", tool_calls=[call]), AIMessage("done")]))
    agent = build_agent(tools, model)
    try:
        agent.invoke({"messages": [{"role": "user", "content": "q"}]})
        out("sync agent.invoke", ok=True)
    except Exception as e:  # noqa: BLE001
        out("sync agent.invoke", ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
    sync_agent_invoke()
    sys.exit(0)
