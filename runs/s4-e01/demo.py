"""Start the docs server as a subprocess over stdio and query the real corpus (run from s4-mcp/)."""
import asyncio, json, sys, time
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

async def main():
    t0 = time.perf_counter()
    async with Client(StdioTransport(command=sys.executable, args=["-m", "docs_mcp.server"])) as c:
        t1 = time.perf_counter()
        tools = await c.list_tools()
        print("tools:", [(t.name, list(t.input_schema["properties"])) for t in tools])
        for q in ["interrupt resume", "checkpointer thread_id", "stream subgraphs"]:
            s = time.perf_counter()
            hits = (await c.call_tool("search_docs", {"query": q, "limit": 3})).data
            print(f"search {q!r} {time.perf_counter()-s:.3f}s ->", [(h["path"], h["score"]) for h in hits])
        page = (await c.call_tool("read_doc", {"path": "langgraph/interrupts.mdx", "max_chars": 300})).data
        print("read_doc:", {k: page[k] for k in ("start", "next_start", "more", "total_chars")})
        bad = await c.call_tool("read_doc", {"path": "../../.gitignore"}, raise_on_error=False)
        print("outside path:", bad.is_error, bad.content[0].text)
    print(f"startup {t1-t0:.2f}s")

asyncio.run(main())
