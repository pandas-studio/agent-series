# S4 docs MCP server — rules for Claude Code

## What this is
A FastMCP server that lets an agent search and read the local doc pages in `../corpus/`
(`corpus/fetch.sh` downloads them): tools `search_docs`/`read_doc`, resources `docs://index` and
`docs://page/{path*}`, prompts `ask_docs` and `summarize_page`. `docs_mcp.agent` is a LangChain agent on the local
model that uses the server's tools through MCP. Episode states are tagged `s4-eNN`.

## Environment
- Python 3.12 with uv. `uv run pytest -q` runs every test, including one that starts the server as a stdio
  subprocess.
- Run the server: `uv run python -m docs_mcp.server` (stdio). `DOCS_ROOT` overrides the doc folder.
- Run the agent: `uv run python -m docs_mcp.agent "<question>"` (needs the local model server; `AGENT_BASE_URL`,
  `AGENT_MODEL` override it). `AGENT_LIVE=1 uv run pytest -q` adds the live test.
- fastmcp 4.x on MCP SDK v2: tool fields are snake_case (`input_schema`, not `inputSchema`).

## Rules
- A tool's docstring and type hints are its contract for the model: keep them short, say what the tool returns,
  and say which tool to call next.
- Tools, resources and prompts never read outside `DOCS_ROOT`. Expected failures raise the primitive's error class:
  `ToolError`, `ResourceError`, `PromptError`. The client gets the message; for resources and prompts FastMCP 4.1.6
  still logs a traceback (expected, not a crash).
- Resource templates that take a page path use `{path*}`: a plain `{path}` matches one URI segment only.
- A prompt returns `str` or `list[Message]`. Plain dicts are refused; an embedded page is
  `Message(EmbeddedResource(...))`, because `Message(dict)` turns the dict into JSON text.
- Over stdio, stdout belongs to the protocol: never print to stdout in server code.
- Claude Code connects through the repository's `.mcp.json` (`claude mcp add --scope project docs -- uv run --directory
  s4-mcp python -m docs_mcp.server`, run from the repository root). Claude Code ignores `cwd` in `.mcp.json` and does not
  read `mcpServers` from `.claude/settings.json`; `uv run --directory` picks the project.
- `claude -p` loads `.mcp.json` servers without the approval prompt, but each tool call still needs permission: pass
  `--allowedTools "mcp__docs__search_docs mcp__docs__read_doc"` (or the user's permission mode decides).
- LangChain agents connect through `langchain.mcp.MCPAdapter` (`langchain[mcp]`, beta in langchain 1.5), not the
  old `langchain_mcp_adapters.MultiServerMCPClient`. `src/docs_mcp/agent.py` passes an `MCPConfig`
  (`{"mcpServers": {"docs": {"command": …, "args": […], "env": {…}}}}`, no `transport` key) and builds the agent
  with `create_agent` after the adapter's `async with` has exited; the tools reconnect on each call.
- The stdio server does not inherit this process's environment: pass `DOCS_ROOT` in the entry's `env`.
- Adapted MCP tools are async only: call the agent with `ainvoke`; `invoke` raises NotImplementedError.
- Keep FastMCP's default stdio `keep_alive`: with `keep_alive=False` every tool call starts a new server process.
- Test through `fastmcp.Client` (in-memory) and keep one stdio subprocess test. Save demo output under
  `../runs/s4-eNN/`.
- Check APIs against the current FastMCP docs (gofastmcp.com) and note differences from the author's book in
  `../runs/s4-eNN/notes.md`. Code and comments in English.

## Episode notes (`../runs/s4-eNN/notes.md`)
- Start with `## Key finding`: 2-3 sentences on the one result the post opens with, including its number.
- List the packages added directly (`uv add …`) apart from the installed dependency versions.
- Then: Commands run, How Claude Code was used, Design decisions, Measurements, Difference from the author's book, Tests.
