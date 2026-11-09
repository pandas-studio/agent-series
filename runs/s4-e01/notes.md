# s4-e01 — MCP concept, first FastMCP server (2026-11-09)

Facts recorded while building this episode. This is source material for the blog pipeline, not post text.

## Key finding
An expected tool failure must be raised as `ToolError`: a `ValueError` from `read_doc` made the server print a full
Rich traceback to stderr, while `ToolError` gave the client the same message and the server one log line. In both
cases the server kept answering. Over 16 doc pages, `search_docs` answered in 0.009–0.039 s per query.

## What was built
- `s4-mcp/src/docs_mcp/server.py`: a FastMCP server named `docs` with two tools over the local doc folder
  (`corpus/`, 16 LangGraph/LangChain pages from langchain-ai/docs, MIT, commit 6c521ae614ef, about 400 KB):
  - `search_docs(query, limit=5)`: word-count ranking, returns path, title, score and one matching line.
  - `read_doc(path, start=0, max_chars=4000)`: reads a page in parts (`next_start`, `more`), because pages are long
    (interrupts.mdx is 61,731 characters).
- No model is involved in this episode. The server only exposes tools; a client (Claude Code in episode 3, a
  LangGraph agent in episode 4) decides when to call them.
- Packages: fastmcp 4.1.6, mcp 2.3.0, pydantic 2.14.3 (`versions.txt`).

## Commands run
- `uv init --lib --name docs-mcp --python 3.12 .` (in s4-mcp/)
- `uv add fastmcp` and `uv add --dev pytest pytest-asyncio`
- `uv run pytest -q` (7 in-memory tests and 1 stdio subprocess test)
- `uv run python -m docs_mcp.server` (runs the server over stdio; `DOCS_ROOT` picks another doc folder)
- `uv run python ../runs/s4-e01/demo.py` (starts the server as a subprocess and queries the real corpus)

## How Claude Code was used
- Claude Code looked up the FastMCP 4.1 docs (Context7, `/prefecthq/fastmcp/v4.1.3`) before writing code: `@mcp.tool`
  without parentheses, `Client(server)` for in-memory tests, `result.data` for the structured return value.
- First test run: 8 passed with a deprecation warning: "Accessing `Tool.inputSchema` is deprecated; MCP SDK v2
  renamed this field to `input_schema`". The test was changed to `input_schema`; written down in CLAUDE.md.
- First stdio demo: a refused path (`../../.gitignore`) made the server print a full Rich traceback to stderr,
  because `read_doc` raised `ValueError`. Changed to `fastmcp.exceptions.ToolError`: the client gets the same
  message ("unknown page: ../../.gitignore", `is_error` true) and the server logs one line.
  The server did not stop in either case (checked 11-09): with `ToolError`, the next `search_docs` call in the same session returned langgraph/interrupts.mdx; a test tool raising `ValueError` also came back as a tool error and the next call answered. Only the stderr log differed (full traceback vs one line).
- The FastMCP 4.1.6 startup banner was printed to stderr; turned off with `mcp.run(show_banner=False)`. Over stdio,
  stdout carries the protocol, so server code never prints to stdout.

## Design decisions
- The tool docstring is what the model reads. `read_doc`'s docstring tells the model where `path` comes from
  (search_docs) and how to continue a long page.
- Type hints become the input schema: `read_doc` has `path` (string, required), `start` and `max_chars`
  (integer, optional) — checked in `test_tool_schema_comes_from_type_hints`.
- Path guard: resolve the path and refuse anything outside DOCS_ROOT, non-.md/.mdx files and missing pages
  (tested with `../../etc/passwd`, `secret.txt`, a missing page).
- Tests: 7 in-memory tests plus 1 test that starts the server as a stdio subprocess, the way Claude Code starts it.

## Measurements (`demo-output.txt`, real corpus, stdio subprocess)
- Server start to ready (connect + initialize): 0.45 s and 0.48 s (2 runs).
- search_docs: 0.009–0.039 s per query over 16 pages.
- Top results: "interrupt resume" → langgraph/interrupts.mdx (score 483); "checkpointer thread_id" →
  langgraph/checkpointers.mdx (159); "stream subgraphs" → langgraph/streaming.mdx (415).

## Difference from the author's book
- Claude Code book chapter 18 (pages 332439–332442) builds a FastMCP server. This episode uses fastmcp 4.1.6 on
  MCP SDK v2, where client-side tool fields are snake_case (`input_schema`).
- Not yet compared line by line with the book's code; do that before writing the post.

## Tests
- 8 passed (`pytest.txt`).
