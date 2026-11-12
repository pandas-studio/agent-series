# S1 research agent — rules for Claude Code

## What this is
A LangGraph research agent that answers from the local doc pages in `../corpus/`: episode 2's routed graph
(`graph.py`), and from episode 3 a tool-calling agent (`agent.py`, tool `search_docs` in `tools.py`).
Each blog episode adds one capability. The episode's state is tagged `s1-eNN`.

## Environment
- Model: OpenAI-compatible server, default vLLM at http://localhost:8000/v1 (Qwen3.6). Settings come only from
  `AGENT_*` environment variables (`src/research_agent/config.py`). Do not hard-code another URL or key.
- Thinking is off by default (`AGENT_THINKING=0`). Turn it on only for a step that needs it, and measure the cost.
- Python 3.12 with uv. Run `uv run pytest -q` for unit tests; `AGENT_LIVE=1 uv run pytest -q -m live` for tests
  that call the model.

## Rules
- Every model-facing function takes the model as a parameter, so unit tests can pass a fake
  (`GenericFakeChatModel`) and run offline.
- Structured output on the local vLLM: `with_structured_output(..., method="json_schema")`. With
  `function_calling` (a forced tool choice) and `json_mode` the local Qwen returned nothing usable
  (`../runs/s1-e02/route-check.jsonl.txt`).
- Tool calls on the local vLLM: only `tool_choice="auto"` returns a tool call; `"required"` and a named tool return
  none (`../runs/s1-e03/force-tool-check.txt`). With auto, the model often answers from memory instead of searching,
  even when the system prompt says to search first. A step that must use a tool is enforced by the graph
  (`build_graph(guard=True)`: router + one reminder), not by the prompt.
- Thinking on with tools can keep generating without stopping; give such calls a timeout and count the timeouts.
- Doc pages pull their code in as MDX snippet components that `corpus/` lacks; `docs.page_text` strips the imports
  and the `<Component />` lines before searching.
- With a checkpointer the thread keeps every turn. Anything that belongs to one turn (model calls, the reminder,
  "searched yet", the question to route) is computed from the messages since the last user question
  (`agent.this_turn`), never kept in a state field that the next turn would inherit.
- SQLite checkpoints: `SqliteSaver.from_conn_string(path)` (`chat.open_saver`). `SqliteSaver(sqlite3.connect(path))`
  fails on the first invoke because LangGraph writes checkpoints from a worker thread.
- A change is done only when both the unit tests and the live tests pass. Save the live output under
  `../runs/s1-eNN/`.
- Check APIs against the current official docs (docs.langchain.com), not memory. When the API differs from the
  author's book, note the difference in `../runs/s1-eNN/notes.md`.
- Code and comments in English. Model prompts in English; user-facing sample questions may be Korean.
- Never commit secrets or the downloaded corpus pages.

## Episode notes (`../runs/s1-eNN/notes.md`)
- Start with `## Key finding`: 2-3 sentences on the one result the post opens with, including its number.
- List the packages added directly (`uv add …`) apart from the installed dependency versions.
- Then: Commands run, How Claude Code was used, Design decisions, Measurements, Difference from the author's book, Tests.
