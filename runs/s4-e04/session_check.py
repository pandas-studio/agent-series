"""How many docs-server processes and sessions do MCPAdapter tools use? No model involved.

For each pattern: discover the tools, then call search_docs 5 times and read_doc once, either after the adapter's
`async with` has exited (the docs' default pattern) or inside it (held). Records server starts (lines that
counting_server.py appends), the tool names, discovery and per-call times.

Run from s4-mcp: uv run python ../runs/s4-e04/session_check.py > ../runs/s4-e04/session-check.jsonl.txt
"""

import asyncio
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

warnings.filterwarnings("ignore", message=".*langchain.mcp.*beta.*")
from langchain.mcp import MCPAdapter  # noqa: E402

from docs_mcp import server  # noqa: E402

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "counting_server.py"
STARTS = SCRIPT.with_suffix(".starts")
CALLS = [("search_docs", {"query": "checkpointer thread_id"})] * 5 + [
    ("read_doc", {"path": "langgraph/persistence.mdx", "max_chars": 2000})]


def starts() -> list[int]:
    return [int(line.split()[0]) for line in STARTS.read_text().splitlines()] if STARTS.exists() else []


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def target(pattern: str):
    if pattern.startswith("path"):
        return SCRIPT
    if pattern.startswith("config"):
        return {"mcpServers": {"docs": {"command": sys.executable, "args": [str(SCRIPT)]}}}
    if pattern.startswith("no-keepalive"):
        return Client(PythonStdioTransport(SCRIPT, keep_alive=False))
    if pattern.startswith("memory"):
        return server.mcp
    raise ValueError(pattern)


async def call_all(tools) -> list[float]:
    by_name = {t.name.split("_", 1)[1] if t.name.startswith("docs_") else t.name: t for t in tools}
    times = []
    for name, args in CALLS:
        t0 = time.perf_counter()
        await by_name[name].ainvoke(args)
        times.append(round((time.perf_counter() - t0) * 1000, 1))
    return times


async def run(pattern: str) -> dict:
    STARTS.unlink(missing_ok=True)
    adapter = MCPAdapter(target(pattern))
    held = pattern.endswith("held")
    t0 = time.perf_counter()
    if held:
        async with adapter:
            tools = await adapter.list_tools()
            discover_ms = round((time.perf_counter() - t0) * 1000, 1)
            times = await call_all(tools)
    else:
        async with adapter:
            tools = await adapter.list_tools()
        discover_ms = round((time.perf_counter() - t0) * 1000, 1)
        times = await call_all(tools)
    pids = starts()
    await asyncio.sleep(0.3)
    return {"pattern": pattern, "tools": sorted(t.name for t in tools), "server_starts": len(pids),
            "alive_after_calls": sum(alive(p) for p in pids), "discover_ms": discover_ms,
            "call_ms": times, "call_median_ms": statistics.median(times),
            "mcp_server_meta": (tools[0].metadata or {}).get("mcp", {}).get("server")}


async def main(patterns: list[str]) -> None:
    for p in patterns:
        print(json.dumps(await run(p), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["path-exit", "path-held", "config-exit", "config-held",
                                     "no-keepalive-exit", "no-keepalive-held", "memory-exit"]))
