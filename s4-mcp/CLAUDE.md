# S4 docs MCP server — rules for Claude Code

## What this is
A FastMCP server that lets an agent search and read the local doc pages in `../corpus/`
(`corpus/fetch.sh` downloads them). Episode states are tagged `s4-eNN`.

## Environment
- Python 3.12 with uv. `uv run pytest -q` runs every test, including one that starts the server as a stdio
  subprocess.
- Run the server: `uv run python -m docs_mcp.server` (stdio). `DOCS_ROOT` overrides the doc folder.
- fastmcp 4.x on MCP SDK v2: tool fields are snake_case (`input_schema`, not `inputSchema`).

## Rules
- A tool's docstring and type hints are its contract for the model: keep them short, say what the tool returns,
  and say which tool to call next.
- Tools never read outside `DOCS_ROOT`. Expected failures raise `ToolError` (message to the client, no traceback).
- Over stdio, stdout belongs to the protocol: never print to stdout in server code.
- Test through `fastmcp.Client` (in-memory) and keep one stdio subprocess test. Save demo output under
  `../runs/s4-eNN/`.
- Check APIs against the current FastMCP docs (gofastmcp.com) and note differences from the author's book in
  `../runs/s4-eNN/notes.md`. Code and comments in English.
