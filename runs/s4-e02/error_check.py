"""Which exception class keeps the server log to one line, per MCP primitive (run from s4-mcp/).

Each case runs a tiny FastMCP server in a subprocess over stdio, triggers one expected failure and reports the
client's error message and the number of lines the server wrote to stderr.
Output: ../runs/s4-e02/error-check.txt
"""
import asyncio
import sys
import tempfile
import textwrap

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

SERVER = textwrap.dedent('''
    import sys
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError, ResourceError, PromptError
    mcp = FastMCP("errors")
    exc = {"ToolError": ToolError, "ResourceError": ResourceError, "PromptError": PromptError,
           "ValueError": ValueError}[sys.argv[1]]

    @mcp.tool
    def t(path: str) -> str:
        raise exc(f"unknown page: {path}")

    @mcp.resource("docs://page/{path*}")
    def r(path: str) -> str:
        raise exc(f"unknown page: {path}")

    @mcp.prompt
    def p(path: str) -> str:
        raise exc(f"unknown page: {path}")

    mcp.run(show_banner=False)
''')


async def case(script: str, exc: str, primitive: str) -> str:
    with tempfile.TemporaryFile("w+") as err:
        transport = StdioTransport(command=sys.executable, args=[script, exc], log_file=err)
        async with Client(transport) as c:
            try:
                if primitive == "tool":
                    res = await c.call_tool("t", {"path": "x.mdx"}, raise_on_error=False)
                    msg = f"is_error={res.is_error} {res.content[0].text!r}"
                elif primitive == "resource":
                    await c.read_resource("docs://page/x.mdx")
                    msg = "no error"
                else:
                    await c.get_prompt("p", {"path": "x.mdx"})
                    msg = "no error"
            except Exception as e:  # the client raises for resource and prompt errors
                msg = f"{type(e).__name__} {str(e)!r}"
            alive = (await c.call_tool("t", {"path": "y"}, raise_on_error=False)).is_error  # server still answers
        err.seek(0)
        lines = [ln for ln in err.read().splitlines() if ln.strip()]
    return f"{primitive:8} {exc:13} stderr_lines={len(lines):3} next_call_answered={alive}  client: {msg}"


async def main():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SERVER)
    for primitive in ("tool", "resource", "prompt"):
        for exc in ("ToolError", "ResourceError", "PromptError", "ValueError"):
            print(await case(f.name, exc, primitive), flush=True)


asyncio.run(main())
