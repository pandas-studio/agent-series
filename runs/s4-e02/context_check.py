"""What each MCP primitive puts in front of the model, in Qwen tokens (run from s4-mcp/).

Starts the docs server over stdio, lists the three primitives, and counts tokens with the local vLLM server's
/tokenize endpoint (same tokenizer and chat template as the model). Also checks a single-segment {path} template
against the {path*} one. Output: ../runs/s4-e02/context-check.txt
"""
import asyncio
import json
import os
import sys
import time
import urllib.request

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport

BASE = os.environ.get("AGENT_BASE_URL", "http://localhost:8000/v1").removesuffix("/v1")
MODEL = os.environ.get("AGENT_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")
PAGE = "langgraph/interrupts.mdx"
QUESTION = [{"role": "user", "content": "LangGraph에서 interrupt 후 다시 시작하려면?"}]


def tokens(**body) -> int:
    req = urllib.request.Request(f"{BASE}/tokenize", json.dumps({"model": MODEL, **body}).encode(),
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["count"]


def as_openai(tool) -> dict:
    """The tool definition an OpenAI-style agent sends with every model request."""
    return {"type": "function", "function": {"name": tool.name, "description": tool.description,
                                             "parameters": tool.input_schema}}


async def template_check() -> None:
    probe = FastMCP("probe")

    @probe.resource("one://page/{path}")
    def one(path: str) -> str:
        return f"got {path}"

    @probe.resource("many://page/{path*}")
    def many(path: str) -> str:
        return f"got {path}"

    async with Client(probe) as c:
        for uri in (f"one://page/{PAGE}", "one://page/interrupts.mdx", f"many://page/{PAGE}"):
            try:
                print(f"template {uri:34} -> {(await c.read_resource(uri))[0].text!r}")
            except Exception as e:
                print(f"template {uri:34} -> {type(e).__name__}: {e}")


async def main():
    async with Client(StdioTransport(command=sys.executable, args=["-m", "docs_mcp.server"])) as c:
        tools = await c.list_tools()
        resources = await c.list_resources()
        templates = await c.list_resource_templates()
        prompts = await c.list_prompts()
        print("tools:", [t.name for t in tools])
        print("resources:", [r.uri for r in resources], "templates:", [t.uri_template for t in templates])
        print("prompts:", [(p.name, [a.name for a in p.arguments]) for p in prompts])

        base = tokens(messages=QUESTION)
        with_tools = tokens(messages=QUESTION, tools=[as_openai(t) for t in tools])
        print(f"\nchat request, question only: {base} tokens; with the 2 tool definitions: {with_tools} "
              f"(+{with_tools - base}, sent on every model call of an agent)")
        for t in tools:
            one = tokens(messages=QUESTION, tools=[as_openai(t)])
            print(f"  {t.name}: +{one - base} tokens alone")

        s = time.perf_counter()
        hits = (await c.call_tool("search_docs", {"query": "interrupt resume", "limit": 5})).data
        t_search = time.perf_counter() - s
        s = time.perf_counter()
        part = await c.call_tool("read_doc", {"path": PAGE})
        t_part = time.perf_counter() - s
        s = time.perf_counter()
        whole = (await c.read_resource(f"docs://page/{PAGE}"))[0].text
        t_whole = time.perf_counter() - s
        index = (await c.read_resource("docs://index"))[0].text
        summary = await c.get_prompt("summarize_page", {"path": PAGE})
        ask = await c.get_prompt("ask_docs", {"question": "interrupt 후 재개 방법"})

        part_text = part.content[0].text
        print(f"\ntool result search_docs (5 hits): {tokens(prompt=json.dumps(hits, ensure_ascii=False))} tokens, "
              f"{t_search:.3f} s")
        print(f"tool result read_doc (default max_chars=4000): {len(part_text)} chars, {tokens(prompt=part_text)} "
              f"tokens, {t_part:.3f} s")
        print(f"resource docs://page/{PAGE}: {len(whole)} chars, {tokens(prompt=whole)} tokens, {t_whole:.3f} s")
        print(f"resource docs://index ({len(json.loads(index))} pages): {len(index)} chars, {tokens(prompt=index)} "
              f"tokens")
        kinds = [(m.role, m.content.type) for m in summary.messages]
        embedded = summary.messages[0].content.resource.text
        print(f"prompt summarize_page: messages {kinds}, embedded page {len(embedded)} chars "
              f"({'same' if embedded == whole else 'differs'} as the resource)")
        print(f"prompt ask_docs: messages {[(m.role, m.content.type) for m in ask.messages]}, "
              f"{tokens(prompt=ask.messages[0].content.text)} tokens")
    print()
    await template_check()


asyncio.run(main())
