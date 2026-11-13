# s4-e04 — connecting the docs server to a LangGraph agent with MCPAdapter (2026-11-13)

Facts recorded while building this episode. This is source material for the blog pipeline, not post text.

## Key finding
The current docs say each tool from `MCPAdapter.list_tools()` opens its own session per call, even after the
adapter's `async with` has exited. Over stdio that did not mean a new server process per call: with FastMCP's default
`keep_alive`, one docs-server process served discovery and all later calls (1 start for 6 calls; about 10 ms per
call; 1 start for a whole pass of 24 agent runs). Only with `keep_alive=False` did each call start a new server: 7
starts for 6 calls and about 540 ms per call (same numbers in 2 runs, `session-check-r*.jsonl.txt`). With the
docs-only system prompt the local Qwen used the MCP tools in 24 of 24 runs; with episode s1-e03's prompt, copied
verbatim, it searched in 6 of 24 (`agent-check-r*.jsonl.txt`).

## What was built
- `s4-mcp/src/docs_mcp/agent.py`: a LangChain `create_agent` agent (a LangGraph graph: model node + tool node) on the
  local model, with the docs server's tools from `MCPAdapter`.
  - `server_config(docs_root)`: an MCPConfig `{"mcpServers": {"docs": {"command": sys.executable, "args": ["-m",
    "docs_mcp.server"]}}}`, plus `"env": {"DOCS_ROOT": …}` when a folder is given. No `transport` key: FastMCP infers
    stdio from `command`.
  - `ask(question)`: the only function that opens the adapter. It opens `MCPAdapter(server_config())`, calls
    `list_tools()`, exits the `async with`, then calls `build_agent(tools)` and runs `agent.ainvoke(...)` (the default
    pattern in the LangChain connections page).
  - `build_agent(tools, model)`: one line, `create_agent(model, tools, system_prompt=SYSTEM)`; it does not touch MCP.
  - `make_model()`: `ChatOpenAI` on `http://localhost:8000/v1`, thinking off, `AGENT_*` environment variables (same
    settings as series S1).
  - `uv run python -m docs_mcp.agent "<question>"` prints the tool calls to stderr and the answer to stdout.
- `s4-mcp/tests/test_agent.py`: 9 tests (in-process FastMCP target for most, one stdio test for `env`) and one live
  test (`AGENT_LIVE=1`). The server code did not change.
- `runs/s4-e04/counting_server.py`: the docs server with one extra line that appends its pid to
  `counting_server.starts` on start; the checks launch it to count server processes.
- `runs/s4-e04/session_check.py`, `agent_check.py`, `bridge_check.py`: the measurements below.
- Packages added directly: `uv add "langchain[mcp]" langchain-openai`. Installed (`versions.txt`): langchain 1.5.0,
  langchain-core 1.8.0, langchain-openai 1.7.4, langgraph 1.2.21, fastmcp 4.1.6 (already there), mcp 2.3.0.
  `langchain-mcp-adapters` is not installed.

## Commands run
- `uv add "langchain[mcp]" langchain-openai` (in s4-mcp)
- `uv run python -m docs_mcp.agent "LangGraph에서 checkpointer는 무엇을 저장하나요?"`
- `uv run python ../runs/s4-e04/session_check.py` (twice, `session-check-r1.jsonl.txt`, `session-check-r2.jsonl.txt`)
- `uv run python ../runs/s4-e04/agent_check.py` (twice, `agent-check-r1.jsonl.txt`, `agent-check-r2.jsonl.txt`)
- `uv run python ../runs/s4-e04/bridge_check.py` (`bridge-check.jsonl.txt`)
- `uv run pytest -q` (26 passed, 1 skipped) and `AGENT_LIVE=1 uv run pytest -q` (`pytest.txt`, `pytest-live.txt`)

## How Claude Code was used
- Claude Code read the current LangChain MCP pages through the docs-langchain MCP server before writing code: the
  overview (`langchain[mcp]>=1.4.0`, beta), connections (connection lifecycle, MCPConfig, ClientGroup), tools
  (errors, metadata, elicitation) and the migration guide from `langchain-mcp-adapters`.
- It then read the installed source, `langchain/mcp/adapter.py` and `tools.py`: `MCPAdapter` has only `client` and
  `list_tools()`; every adapted tool is a `StructuredTool` whose coroutine calls the tool through the client.
- The connections page says a run "opens one session per tool call". Claude Code checked FastMCP's
  `StdioTransport` before measuring: `keep_alive` defaults to True ("the subprocess remains active after the
  connection context exits"). To see what that means for an agent, it wrote `counting_server.py` and
  `session_check.py` instead of reasoning from the docs.
- The first agent run (one question) worked on the first try: search_docs once, read_doc three times
  (checkpointers.mdx in 4,000-character parts), answer naming the page path, 10 s.
- `bridge_check.py` first called `ToolNode(...).ainvoke` outside a graph and failed with `ValueError: Missing required
  config key 'N/A' for 'tools'`; it now passes a ToolCall dict to the tool, which returns a ToolMessage as ToolNode
  does.
- A test that set `DOCS_ROOT` with `monkeypatch` and started the server over stdio found nothing: the subprocess does
  not get the parent's environment. Claude Code checked it separately (`bridge-check.jsonl.txt`, last-but-one line)
  and added the `env` entry to `server_config`.
- The first `session_check.py` run printed its results to the terminal only (stdout was not redirected); the two
  saved runs are the second and third. The terminal run had the same starts and similar times (first call after
  exit 98.9 ms, then 10 ms; keep_alive off 518–566 ms).
- Rules went into `s4-mcp/CLAUDE.md`: connect through `MCPAdapter` with an MCPConfig, pass `DOCS_ROOT` in `env`, the
  tools are async only, keep the default `keep_alive`.

## Design decisions
- MCPConfig rather than a `Path` target: the docs server is a package module (`python -m docs_mcp.server`), and the
  MCPConfig entry is the same shape as Claude Code's `.mcp.json` from episode 3 (and the book's dict, minus
  `transport`). `session_check.py` shows both forms behave the same (1 start, ~10 ms per call).
- `sys.executable` as the command, so the server runs in the same uv environment as the agent.
- The agent runs after the adapter context has exited (the docs' default), not inside it: the measurements show no
  extra server start for it.
- `create_agent` instead of the hand-written StateGraph of series S1: this episode is about the MCP bridge; S1-3
  already compared the two builds on the same tool.

## Measurements
### Sessions and processes (`session-check-r1.jsonl.txt`, `session-check-r2.jsonl.txt`, no model)
Discover the tools, then call search_docs 5 times and read_doc once; "exit" = calls after the `async with` has
exited, "held" = calls inside it. Numbers are r1 / r2.
- Path target (`counting_server.py`), exit: 1 server start; discovery 437 / 427 ms (starting the server); calls
  62.9 / 62.5 ms for the first, then 10–11 ms; the server process was still alive after the calls.
- Path target, held: 1 start; calls 7–12 ms.
- MCPConfig (same script), exit and held: 1 start; calls 10–13 ms.
- `Client(PythonStdioTransport(script, keep_alive=False))`, exit: 7 starts (discovery + one per call); every call
  515–568 ms (median 544 / 538 ms); no server process left afterwards.
- keep_alive=False, held: 1 start; calls 10–13 ms; the process ended when the context exited.
- In-process `FastMCP` object as the target: 0 starts; discovery 3 ms; calls 8–14 ms.
- In the agent check, each pass (24 agent runs; 102 and 108 tool calls) had 1 server start.

### The local Qwen with MCP tools (`agent-check-r1.jsonl.txt`, `agent-check-r2.jsonl.txt`)
s1-e03's 4 docs questions, 3 times each, per prompt, 2 runs; tools discovered once per run, agent after the
adapter exit.
- Prompt "s4" (`docs_mcp.agent.SYSTEM`: search with search_docs, read the best page with read_doc, answer in Korean
  with the page path): searched 12/12 and 12/12; read_doc in every run; expected page path in the answer 11/12 and
  10/12; mean model calls 7.5 and 7.92 (min 4, max 13); mean 15.5 s and 14.5 s per question; no tool errors.
  It read pages in 4,000-character parts, often to the end (57 and 59 read_doc calls in 12 runs), as the
  `read_doc` docstring says ("continue with start=next_start while more is true").
- Prompt "s1" (the SYSTEM prompt of series S1 episode 3, `s1-e03`, another series; not S4's previous episode, which
  was about Claude Code. Copied verbatim: search first for LangGraph/LangChain questions, answer other questions
  directly, 3-5 sentences): searched 3/12 and 3/12 (5 of the 6 were the conditional-edges question); expected page
  path in the answer 0/12 and 0/12, also in the runs that searched; mean model calls 3.5 and 2.58; mean 4.6 s and
  3.8 s. In s1-e03 the same prompt with one local `@tool search_docs` (no read tool) searched 2/24 with
  `create_agent`.
- Both prompts tell the model to search before answering LangGraph/LangChain questions ("call search_docs first"
  in s1, "Search the docs with search_docs" in s4). The s1 prompt also says to answer other questions directly
  without tools and to answer in 3-5 sentences; the s4 prompt adds "read the best page with read_doc".
- What differs from s1-e03 is the prompt and the tool set (two MCP tools with their docstrings instead of one local
  tool); these runs do not separate the two, so they do not show which change made the model search. The records do
  not say why the model skipped the search.

### Bridge facts (`bridge-check.jsonl.txt`, no model)
- `list_tools()` returns the 2 tools only. The server also has resource `docs://index`, template
  `docs://page/{path*}` and prompts `ask_docs`, `summarize_page` (episode 2); the adapter has no method for them
  (`MCPAdapter` public names: `client`, `list_tools`). `adapter.client.list_resources()` / `list_prompts()` reach
  them through the FastMCP client.
- An adapted tool: `StructuredTool`, description = the server's docstring, args = the MCP input schema, response
  format `content_and_artifact`, metadata `{"mcp": {"tool": {"_meta": {"fastmcp": {"tags": []}}}, "server":
  {"name": "docs", "version": "4.1.6"}}}` (the version is FastMCP's; the server sets none).
- Two entries in one MCPConfig (`docs`, `notes`): tools `docs_read_doc`, `docs_search_docs`, `notes_read_doc`,
  `notes_search_docs`. One entry: no prefix (`search_docs`, `read_doc`). With two entries FastMCP logs "Proxy
  detected connected client - reusing existing session for all requests" (INFO).
- `read_doc` with an unknown page (the server raises `ToolError`): ToolMessage `status="error"`, content `unknown
  page: langgraph/no-such-page.mdx`; the run continues. A successful call has the JSON text as content and
  `artifact = {"structured_content": …}`.
- `tool.invoke(...)` and `agent.invoke(...)` (sync): `NotImplementedError: StructuredTool does not support sync
  invocation.` Only `ainvoke` works.
- `MCPAdapter("src/docs_mcp/server.py")`: `ValueError: MCP target 'src/docs_mcp/server.py' is not a valid URL. To run
  a local MCP server over stdio, ask for it explicitly with Path('server.py'), a fastmcp transport, or an MCPConfig`.
- `DOCS_ROOT` set only in the parent process: the stdio server searched its default folder (`"no env entry":
  {"result": []}` for a word only in the temporary folder); with `"env": {"DOCS_ROOT": …}` it found the page.

## Difference from the author's book
Deep Agents book 2.7 (page 379471) uses the standalone package `langchain-mcp-adapters`. In langchain 1.5.0 MCP is in
`langchain.mcp` (`langchain[mcp]`); the migration guide marks the old API deprecated.
- Package: page 379471 says "Deep Agents는 `langchain-mcp-adapters` 패키지로 MCP 서버의 도구를 일반 LangChain 도구로
  변환해 붙입니다." and installs `uv add langchain-mcp-adapters`; this episode used `uv add "langchain[mcp]"`.
- Client: page 379471 says "`MultiServerMCPClient`에 서버별 연결 설정을 딕셔너리로 전달합니다." and
  "`await client.get_tools()`로 모든 서버의 도구를 한 번에 가져옵니다."; now
  `async with MCPAdapter(config) as adapter: tools = await adapter.list_tools()`.
- Transport key: page 379471 says "각 서버는 트랜스포트(`stdio` / `streamable_http` / `sse`)에 따라 필드가 다릅니다." and
  writes `"transport": "stdio"` / `"streamable_http"` in each entry. The
  MCPConfig entry has no transport key; `command` means stdio, `url` means HTTP.
- One server's tools: page 379471 uses `await client.get_tools(server_name="filesystem")`. `list_tools()` has no
  server argument; the migration guide says to point an adapter at that server. With two entries in one MCPConfig
  the names are prefixed (`docs_search_docs`), so the server is in the tool name.
- Sessions: the TIP on page 379471 says 0.3.x replaced `async with client.session(...)` with calling
  `await client.get_tools()` directly. `MCPAdapter` exposes no session at all; each tool call opens one, and over stdio
  the server process stays up between calls (Measurements).
- `env`: page 379471 shows `"env": {"GITHUB_TOKEN": …}` for a stdio server. It is also needed for any setting the
  server reads from the environment: the subprocess did not see the parent's `DOCS_ROOT`.
- Async: page 379471 says `get_tools()` is a coroutine and to run the agent with `ainvoke`. Still true; the adapted
  tools also fail on sync `invoke` (`NotImplementedError`).
- Deployment: page 379471's TIP says stdio servers are refused in LangSmith deployments. Not tested here (local only).

## Tests
- 26 passed, 1 skipped (`pytest.txt`): episode 3's 17 plus 9 in `test_agent.py`.
- Live (`AGENT_LIVE=1 uv run pytest -q tests/test_agent.py -k live`, `pytest-live.txt`): 1 passed in 12.18 s.
