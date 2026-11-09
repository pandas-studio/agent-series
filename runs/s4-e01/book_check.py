"""Book 18-1/18-2 patterns against the installed fastmcp (s4-e01 notes, "Difference from the author's book").

Run from s4-mcp/: uv run python ../runs/s4-e01/book_check.py
"""
import asyncio
import inspect
import json

import fastmcp
from fastmcp import Client, FastMCP
from fastmcp.server.server import Transport

from docs_mcp import server

book = FastMCP("my-server")


@book.tool()
def greet(name: str) -> str:
    """Greets the user.

    Args:
        name: the name to greet

    Returns:
        greeting string
    """
    return f"hi {name}"


def plain(x: int) -> int:
    """Returns x."""
    return x


book.tool()(plain)   # the book registers functions from another module this way


async def main():
    print("fastmcp", fastmcp.__version__)
    print("transports:", Transport.__args__)
    async with Client(book) as c:
        tools = {t.name: t for t in await c.list_tools()}
        print("book-style tools:", sorted(tools), "calls:", (await c.call_tool("greet", {"name": "a"})).data,
              (await c.call_tool("plain", {"x": 3})).data)
        print("greet as the client sees it:", json.dumps(tools["greet"].model_dump(exclude_none=True), ensure_ascii=False))
    async with Client(server.mcp) as c:
        for t in await c.list_tools():
            print(t.name, "description:", repr(t.description))
            print(t.name, "parameter descriptions:", {k: v.get("description") for k, v in t.input_schema["properties"].items()})


asyncio.run(main())
