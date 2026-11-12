# s4-e03 — connecting the docs server to Claude Code (2026-11-12)

Facts recorded while building this episode. This is source material for the blog pipeline, not post text.

## Key finding
The book's way of registering the server, `mcpServers` in `.claude/settings.json` with a `cwd`, is not read by Claude
Code 2.1.322 (`claude mcp list` shows no such server); `claude mcp add --scope project` writes `.mcp.json`, which is.
Once connected, the docs server's two tools cost little context: they add 124 input tokens to Claude Code's first
request with tool search (the default) and 1,169 tokens with `ENABLE_TOOL_SEARCH=false` (same numbers in 2 runs,
`claude-code-check*.jsonl.txt`). In `claude -p` the `.mcp.json` server is started without the approval prompt that an
interactive session shows, but in the default permission mode each tool call is denied unless `--allowedTools` names it.

## What was built
- `.mcp.json` at the repository root (project scope), written by `claude mcp add`:
  `{"mcpServers": {"docs": {"type": "stdio", "command": "uv", "args": ["run", "--directory", "s4-mcp", "python", "-m",
  "docs_mcp.server"], "env": {}}}}`. No server code changed in this episode.
- `s4-mcp/tests/test_server.py::test_project_mcp_json_starts_the_server`: reads `.mcp.json`, starts its command from
  the repository root over stdio (as Claude Code does) and checks the two tools.
- `runs/s4-e03/claude_code_check.py`: runs `claude -p` with `--output-format stream-json` for 9 cases and records the
  docs server status, offered `mcp__docs__*` tools, tool calls, permission denials, input tokens of the first model
  request, turns and time. Only the docs server is recorded; this machine's other MCP servers are left out.
- No packages added. Versions (`versions.txt`): fastmcp 4.1.6, mcp 2.3.0; Claude Code 2.1.322, model in the runs
  claude-sonnet-5-5.

## Commands run
- `claude mcp add --scope project docs -- uv run --directory s4-mcp python -m docs_mcp.server` (in the repository root)
- `claude mcp list` and `claude mcp get docs` (`mcp-add.txt`)
- `python3 runs/s4-e03/claude_code_check.py` (cases A-H, `claude-code-check.jsonl.txt`) and
  `python3 runs/s4-e03/claude_code_check.py I A F G H` (second run, `claude-code-check-r2.jsonl.txt`)
- `claude -p … --mcp-config .mcp.json --strict-mcp-config` in a scratch project for the book's settings.json form
  (`book-check.txt`)
- `uv run pytest -q` in s4-mcp (17 passed)

## How Claude Code was used
- Claude Code read the current Claude Code MCP page (code.claude.com/docs/en/mcp) before registering: project scope
  writes `.mcp.json`; a project server stays "⏸ Pending approval" until approved in an interactive session; `claude -p`
  runs load project-scoped servers without asking; tool search (deferred tool definitions) is on by default; MCP
  prompts are listed as `/servername:promptname (MCP)` and `/mcp__servername__promptname` also works; text results
  over 50,000 characters are saved to a file.
- After `claude mcp add`, `claude mcp list` showed `docs: … - ⏸ Pending approval (run \`claude\` to approve)`. The
  repository folder had never been trusted (no entry for it in the Claude Code project list).
- First headless runs: the tool calls went through without `--allowedTools`. Claude Code checked this machine's user
  settings: `permissions.defaultMode` is `auto`, whose classifier approved the calls. Those runs are kept as
  `claude-code-check-auto-mode.jsonl.txt`; all later runs pass `--permission-mode default`, the mode a reader starts
  with.
- The first token comparison turned tool search off with the docs server (39,293 input tokens) against tool search on
  without it (22,085). That difference (17,208) also contains Claude Code's own deferred tools, so Claude Code added
  case I (no MCP server, tool search off: 38,124) and compared like with like.
- The book's settings.json form was tried in an empty scratch project (`book-check.txt`): not listed, not loaded; the
  same entry in `.mcp.json` failed to connect with or without `cwd`, and connected once the args used
  `uv run --directory <absolute path to s4-mcp>`.
- Rules went into `s4-mcp/CLAUDE.md`: connect through `.mcp.json`, `cwd` is ignored, `claude -p` needs
  `--allowedTools` for the tools.

## Design decisions
- Project scope (`.mcp.json` in the repository) so a reader who clones the repository gets the same server entry;
  `--directory s4-mcp` instead of an absolute path, because Claude Code starts project servers from the repository
  root.
- The checks use `--strict-mcp-config` with `--mcp-config .mcp.json`, so only the docs server is in the session and the
  token numbers are not mixed with other servers. Case A runs without these flags to show what plain `claude -p` does.
- One run per case (they use the Claude subscription), plus a second run of the token and approval cases.

## Measurements (`claude-code-check.jsonl.txt`, `claude-code-check-r2.jsonl.txt`, permission mode default)
- A, plain `claude -p` in the repository: the docs server connected (`source: project`) although `claude mcp list`
  said Pending approval; Claude called `search_docs`, the call was denied ("Claude requested permissions to use
  mcp__docs__search_docs, but you haven't granted it yet"), and Claude did not answer from memory; it said it could
  not search and asked for permission. 12.6 s and 13.6 s (2 runs).
- B, `--mcp-config .mcp.json --strict-mcp-config`, no allowed tools: the same denial, 10.0 s.
- C, B plus `--allowedTools "mcp__docs__search_docs mcp__docs__read_doc"`: `search_docs` once, `read_doc` three times
  (checkpointers.mdx, 4,000 characters at a time), answer with the page path, 6 turns, 17.3 s.
- D, prompt starting with `@docs:docs://page/langgraph/checkpointers.mdx` (no allowed tools): the resource was not
  attached to the prompt (first request 22,235 tokens, the same as without it); Claude read it with
  `ReadMcpResourceTool`, the 57.9 KB result was saved to a file ("Output too large (57.9KB). Full output saved to …"),
  a Bash command to read the file was denied (needs approval), and `Read` read the saved file; summary in 6 turns,
  21.7 s. Interactive `@` autocomplete was not tested.
- E, `/mcp__docs__ask_docs LangGraph에서 checkpointer는 무엇을 저장하나요?` with the allowed tools: the MCP prompt ran in
  `claude -p`; Claude followed it (search once, read 4 times), 7 turns, 26.0 s. The session's slash commands included
  `mcp__docs__ask_docs` and `mcp__docs__summarize_page`.
- First-request input tokens (same in both runs): no MCP server 22,085; docs server with tool search 22,209 (+124);
  docs server with `ENABLE_TOOL_SEARCH=false` 39,293; no server with `ENABLE_TOOL_SEARCH=false` 38,124. The docs
  server's two tool definitions: +124 tokens deferred, +1,169 loaded. (Episode 2 measured the same two definitions on
  the local Qwen chat template at +446 tokens; different tokenizers and templates.)

## Difference from the author's book
- Registration file: Claude Code book 18-4 (page 332442) asks Claude Code to add the server to `.claude/settings.json`
  as `"mcpServers": {"github-tools": {"type": "stdio", "command": "uv", "args": ["run", "python", "server.py"], "cwd":
  …}}`. On Claude Code 2.1.322 `mcpServers` in `.claude/settings.json` is not read (`book-check.txt`). This episode uses
  `claude mcp add --scope project`, which writes `.mcp.json`.
- `cwd`: page 332442 says `cwd` must be the source directory's absolute path. In `.mcp.json` the `cwd` key did not make
  the server start in that directory (status failed); `uv run --directory s4-mcp` did.
- Connection check: page 332442 restarts Claude Code and runs `/mcp` (the server shows Green). This episode checked with
  `claude mcp list` / `claude mcp get docs` (Pending approval until approved interactively) and the `system/init` event
  of `claude -p` (status connected).
- Approval: Claude Code book 10-3 (page 329900) says project-scope servers from `.mcp.json` show an approval prompt on
  first use. That holds for interactive sessions; in `claude -p` the server was loaded without a prompt (A), as the
  current docs say.
- Scopes: page 329900 lists local, project and user scopes with `claude mcp add --scope`; the same three scopes exist in
  2.1.322 (`claude mcp add --help`). This episode used only project scope and did not touch the user or local files.
- Context cost: book 10-4 (page 329905) says that when MCP tool definitions exceed 10% of the context window, Claude
  Code switches to on-demand loading (`ENABLE_TOOL_SEARCH`). In 2.1.322 tool search is on from the start: the docs
  server added 124 tokens deferred and 1,169 with `ENABLE_TOOL_SEARCH=false`.
- MCP prompts: page 329905 writes them as `/mcp__서버이름__명령어이름`; `/mcp__docs__ask_docs` worked in `claude -p` (E).
  The current docs list them as `/servername:promptname (MCP)` in the `/` menu.
- MCP resources: page 329905 references them as `@서버이름:프로토콜://리소스/경로`. In `claude -p` the
  `@docs:docs://page/…` text was not attached; Claude fetched the resource with `ReadMcpResourceTool` (D).

## Tests
- 17 passed (`pytest.txt`): episode 2's 16 plus `test_project_mcp_json_starts_the_server`.
