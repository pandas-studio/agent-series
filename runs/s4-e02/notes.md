# s4-e02 — tools, resources, prompts (2026-11-10)

Facts recorded while building this episode. This is source material for the blog pipeline, not post text.

## Key finding
The same doc page costs very different context depending on the primitive that carries it: `read_doc` (a tool)
returned the first 4,153 characters of langgraph/interrupts.mdx as 990 Qwen tokens, while the resource
`docs://page/langgraph/interrupts.mdx` returned the whole page, 61,731 characters and 15,217 tokens (about 15 times
more). The two tool definitions themselves add 446 tokens to every model request of an agent; the resources and
prompts add nothing until the client or the user attaches one (`context-check.txt`).

## What was built
- `s4-mcp/src/docs_mcp/server.py` keeps episode 1's two tools and adds the other two MCP server primitives:
  - resource `docs://index` (JSON): every page's path, title, size in characters and resource URI (16 pages).
  - resource template `docs://page/{path*}` (text/markdown): the whole text of one page.
  - prompt `ask_docs(question)`: one user message that tells the model to call `search_docs`, then `read_doc`,
    answer in Korean and list the page paths.
  - prompt `summarize_page(path)`: two user messages, the page embedded as a resource and a request for a 5-point
    Korean summary.
- Who decides when each one is used (MCP spec 2026-07-28, server overview): prompts are user-controlled (slash
  commands, menu options), resources are application-controlled (attached and managed by the client), tools are
  model-controlled (the model calls them).
- No new packages. `mcp.types.EmbeddedResource` and `TextResourceContents` come from `mcp`, already installed as a
  dependency of fastmcp.
- Installed versions (`versions.txt`): fastmcp 4.1.6, mcp 2.3.0, pydantic 2.14.3; token counts from vLLM 0.21.0
  `/tokenize` with the served model nvidia/Qwen3.6-35B-A3B-NVFP4.

## Commands run
- `uv run pytest -q` (16 passed: 15 in-memory, 1 stdio subprocess)
- `uv run python ../runs/s4-e02/context_check.py` (output `context-check.txt`; needs the local model server for
  `/tokenize`)
- `uv run python ../runs/s4-e02/error_check.py` (output `error-check.txt`)

## How Claude Code was used
- Claude Code looked up the FastMCP 4.1 docs first (Context7, `/prefecthq/fastmcp/v4.1.3`): `@mcp.resource` with a
  URI, `{param*}` wildcard parameters in resource templates, `@mcp.prompt`, `Client.read_resource` and
  `Client.get_prompt` for tests.
- First version of `summarize_page` returned plain dicts (`{"role": "user", "content": {"type": "resource", ...}}`),
  as in the docstring example of the FastMCP API reference page for `prompt`. On 4.1.6 `get_prompt` failed:
  "messages[0] must be Message or str, got dict. Use Message(...) to wrap the value." Claude Code then checked what
  `Message(dict)` does: it serializes the dict to JSON and sends it as text content (`content.type` was "text"), so
  the page would arrive as a JSON string, not as a resource. The prompt now returns
  `Message(EmbeddedResource(type="resource", resource=TextResourceContents(...)))`; the test checks that the first
  message's content type is "resource". `TextResourceContents` takes `mime_type` (snake_case, MCP SDK v2).
- First version of the page resource raised `ToolError` for a refused path (episode 1's guard). The client got
  "unknown page: …", but the server log showed a full traceback, unlike the one-line log for the same error from a
  tool. Claude Code measured all combinations (`error_check.py`: 3 primitives × 4 exception classes, stdio
  subprocess) and read the cause in fastmcp 4.1.6 `server/server.py`: a `FastMCPError` from a tool is logged with
  `exc_info=False`, from a resource or prompt with `exc_info=True`. The resource now raises `ResourceError` and the
  prompt `PromptError` (the right names; the log is the same).
- The URI `docs://page/../../.gitignore` reached the resource function as `.gitignore` (the dot segments were gone);
  it was refused because it is not a .md/.mdx page.
- Rules went into `s4-mcp/CLAUDE.md`: error class per primitive, `{path*}` for page paths, prompts return
  `list[Message]` with `EmbeddedResource` for a page.

## Design decisions
- The page resource returns the whole page on purpose: a resource is attached by the client or the user, who
  chooses the page, so it does not need the paging that `read_doc` gives the model (`start`, `next_start`, `more`).
  The cost is the size: interrupts.mdx is 15,217 tokens.
- `docs://index` is a resource, not a tool: a client can show it as a page list (a picker) without a model call.
- `ask_docs` does not embed any page; it tells the model which tools to use. `summarize_page` embeds the page, so
  the model needs no tool call. The two prompts show the two ways a prompt can bring in context.
- `{path*}` in the template, because page paths have a folder (`langgraph/interrupts.mdx`).

## Measurements (`context-check.txt`, real corpus, stdio subprocess, Qwen tokenizer)
- Chat request with only the question: 19 tokens. With the 2 tool definitions: 465 tokens (+446). `search_docs`
  alone +316, `read_doc` alone +331 (each includes the chat template's tool preamble, so the two do not add up).
- Tool results: `search_docs` (5 hits) 279 tokens, 0.076 s; `read_doc` with the default `max_chars=4000` 4,153
  characters (JSON with the paging fields), 990 tokens, 0.004 s.
- Resources: `docs://page/langgraph/interrupts.mdx` 61,731 characters, 15,217 tokens, 0.003 s; `docs://index`
  (16 pages) 2,130 characters, 709 tokens.
- Prompts: `ask_docs` one text message, 72 tokens; `summarize_page` messages (user, resource) + (user, text), the
  embedded page identical to the resource (61,731 characters).
- Template check: a plain `{path}` template returned "Resource not found" for `one://page/langgraph/interrupts.mdx`
  and matched only `one://page/interrupts.mdx`; `{path*}` matched `many://page/langgraph/interrupts.mdx`.
- Error log per primitive (`error-check.txt`): `ToolError`/`ResourceError`/`PromptError` from a tool: 4 stderr
  lines (start-up lines and "Error calling tool 't'", no traceback), client message "unknown page: x.mdx". The same classes from a resource or a
  prompt: 29 stderr lines (traceback), client message the same. `ValueError`: 52 (tool), 53 (resource), 96 (prompt)
  lines, and the client message gets a prefix ("Error calling tool 't': …", "Error reading resource …", "Error
  rendering prompt 'p': …"). In all 12 cases the next call in the same session was answered.

## Difference from the author's book
Claude Code book 18-2 (page 332440) builds the GitHub server's tools and tests; 10-4 (page 329905) shows MCP
resources, MCP prompts and the context cost from the Claude Code side. The book has no server code for resources or
prompts.
- Context cost of tool results: page 332440 says that returning the whole GitHub API response uses Claude's context
  window unnecessarily, so a tool should pick only the core fields. Measured here: the same page is 990 tokens
  through `read_doc` (4,000 characters at a time) and 15,217 tokens as the whole-page resource.
- Context cost of tool definitions: page 329905 says that the more MCP servers are registered, the more tool
  definitions are added to the context window and use tokens. Measured here: 2 tool definitions = 446 tokens on
  every request with the Qwen chat template.
- Resources and prompts: page 329905 shows how Claude Code uses them (`@서버이름:프로토콜://리소스/경로`, for example
  `@github:issue://42`; prompts as `/mcp__서버이름__명령어이름`) but not how a server defines them. This episode
  defines them with `@mcp.resource` and `@mcp.prompt`. By that format this server's resource would be
  `@docs:docs://page/langgraph/interrupts.mdx` and the prompt `/mcp__docs__ask_docs`; not checked in Claude Code in
  this episode (episode 3 connects the server to Claude Code).
- Errors: page 332440 raises `ToolError` so that Claude can pass the error message to the user. That holds for tools
  here too. For resources and prompts the book has no example; on fastmcp 4.1.6 `ResourceError`/`PromptError` give
  the client the same clean message, but the server log keeps the traceback.
- Plan Mode: page 332440 designs the tool specs in Plan Mode first (return fields, error handling). This episode did
  not use Plan Mode; Claude Code read the FastMCP docs first and the tests fixed the return shapes.

## Tests
- 16 passed (`pytest.txt`): episode 1's 8 plus 8 for resources and prompts (listing, index, nested page path,
  3 refused paths, embedded page in `summarize_page`, tool names in `ask_docs`), and the stdio test now also reads a
  page resource.
