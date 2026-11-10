# s1-e03 — tool calling: @tool, ToolNode, create_agent (2026-11-10)

Facts recorded while building this episode. This is source material for the blog pipeline, not post text.

## Key finding
Left to decide on its own, the local Qwen rarely called the search tool: on the 4 docs questions, run 6 times each
(3 measurements × 2 runs), the hand-written StateGraph agent searched 2 of 24 times and `create_agent` 2 of 24 times,
although the system prompt says to call `search_docs` first; it answered from memory instead. Forcing the call does
not work on this server: `tool_choice="required"` and a named tool returned no tool call at all (4 of 4 each). A graph
guard (episode 2's router + one reminder when a docs question is answered without a search) made it search 16 of 16
times, at about one more model call (mean 2.06 calls, 5.25 s vs 2.65 s per question, `tool-check.jsonl.txt`).

## What was built
- `src/research_agent/tools.py`: `search_docs(query: str) -> str` with `@tool` (langchain_core). It wraps episode 2's
  `docs.search` and returns the top 3 pages as "[path] title" plus an excerpt. Its docstring says the pages are in
  English and asks for English keywords and API names. `with_description()` makes the same tool with another
  description (used for the "Search the docs." comparison).
- `src/research_agent/agent.py`:
  - `build_graph()`: `StateGraph(AgentState)` (MessagesState + `model_calls`, `route`, `reminded`), node `agent`
    (`model.bind_tools(tools)` with the system prompt), node `tools` (`ToolNode`), a conditional edge `after_agent`
    (tool_calls → tools, else END) and `tools → agent`, as in the LangGraph quickstart.
  - `build_graph(guard=True)`: adds `classify` (episode 2's `Route` router, `json_schema`) before the agent and a
    `remind` node: a docs question answered without any search gets one HumanMessage ("Search the documentation with
    search_docs before answering, …") and goes back to the agent. At most one reminder.
  - Model-call cap `MAX_MODEL_CALLS = 4`: after 4 calls the pending tool calls run, then node `final` calls the model
    once without tools ("Do not search again. Answer now from the pages above").
  - `build_prebuilt()`: `langchain.agents.create_agent(model, [search_docs], system_prompt=SYSTEM)`, the same model,
    tool and prompt.
- `src/research_agent/tool_check.py`: episode 2's 8 labelled questions (4 docs, 4 direct), 2 runs per variant:
  StateGraph, create_agent, StateGraph with the vague description, StateGraph with thinking on, StateGraph + guard.
- `src/research_agent/docs.py`: `page_text()` removes the MDX snippet imports (`import X from '/snippets/…'`) and the
  `<X />` component lines before scoring and excerpting.
- Packages added directly: `uv add langchain` (for `create_agent`; langchain 1.4.9). `langgraph-prebuilt` 1.1.1
  (`ToolNode`) was already installed as a dependency of langgraph.
- Installed versions (`versions.txt`): langgraph 1.2.20, langgraph-prebuilt 1.1.1, langchain 1.4.9, langchain-core
  1.7.6, langchain-openai 1.7.4, openai 3.46.0, pydantic 2.14.3; vLLM 0.21.0 with `--enable-auto-tool-choice
  --tool-call-parser qwen3_coder --reasoning-parser qwen3`.

## Commands run
- `uv add langchain`
- `uv run python -m research_agent.agent "StateGraph에서 조건부 엣지는 어떻게 추가하나요?"` (also `--guard`, `--prebuilt`,
  and the 404 question; twice each, `agent-run.jsonl.txt`)
- `TOOL_CHECK_THINKING=0 uv run python -m research_agent.tool_check > ../runs/s1-e03/tool-check.jsonl.txt`
- `uv run python ../runs/s1-e03/force_tool_check.py` (output `force-tool-check.txt`)
- `uv run python ../runs/s1-e03/thinking_runaway_check.py` (outputs `thinking-runaway-check-*.txt`)
- `uv run pytest -q` and `AGENT_LIVE=1 uv run pytest -q -m live`

## How Claude Code was used
- Claude Code read the current docs first (docs.langchain.com through the docs MCP server): the LangGraph quickstart
  writes the tool loop with `StateGraph(MessagesState)`, its own `should_continue` function and `tool_calls`;
  `ToolNode` comes from `langgraph.prebuilt`; `create_agent` from `langchain.agents`. `tools_condition` is not on
  these pages any more, so the routing function is written by hand.
- First smoke run: for "StateGraph에서 조건부 엣지는 어떻게 추가하나요?" both builds answered in one model call without
  searching (2.04 s and 1.85 s), although the system prompt says to call search_docs first. Claude Code then measured
  how often the model searches (`tool_check.py`) instead of changing the prompt.
- It tried to force the call (`force_tool_check.py`, first model call only, thinking off): `tool_choice="auto"`
  searched 1 of 4 docs questions; `"required"` generated 23-26 tokens with finish reason "tool_calls" but returned no
  tool call (4 of 4); the named tool generated 15-17 tokens, finish "stop", no tool call (4 of 4). Same server
  behaviour as episode 2 (`tool_choice` other than auto returns nothing). So the guard was put in the graph.
- With thinking on, 3 of 16 runs timed out at 300 s. A traced run (interrupt question) showed the third model call
  taking 420 s for 458 output tokens: two 200 s timeouts and a retry, hidden by ChatOpenAI's default retries. The
  vLLM log showed one request generating at about 25 tokens/s the whole time. `thinking_runaway_check.py` sends the
  same agent step 6 times without retries and stops reading at 180 s:
  - before the MDX fix (input after two searches): 5 of 6 and 3 of 6 attempts still generating at 180 s (about
    4,400 tokens each); the others finished in 9.9-38.8 s.
  - after the MDX fix (input after one search): 4 of 6 and 1 of 6 still generating at 180 s (8,868-10,962 tokens);
    the others finished in 2.1-33.6 s. The inputs differ (one search instead of two), so this does not show that the
    fix reduced it.
  - The reasoning text of a non-stopping attempt repeats itself: "Let's search for `Command` in the docs." over and
    over, or "I will output. One detail: … Correct." for each word, or the answer is written in the reasoning and
    then "One last check" again. No tool call and no answer comes out.
- The runaway reasoning searched for names like `LanggraphInterruptsResumeV2Py`. Claude Code found where they come
  from: the doc pages import their code examples as MDX components (`import LanggraphInterruptsResumeV2Py from
  '/snippets/code-samples/…'`; 11 such imports in interrupts.mdx, 51 in graph-api-control-flow.mdx), corpus/ has no
  snippet files, and search excerpts began at these import lines. `docs.page_text` now removes them; excerpts start
  at the page text (interrupts.mdx: 61,731 → 60,032 characters).
- A run that hit the model-call cap ended on a tool call, and its answer was an empty string (4 of 128 runs: 2
  without the guard with the interrupt question before the MDX fix, 2 with the guard and the conditional-edge
  question after it,
  `tool-check-before.jsonl.txt`, `tool-check-nofinal.jsonl.txt`). The `final` node fixed it (0 empty answers in
  `tool-check.jsonl.txt`).
- Rules went into `s1-langgraph/CLAUDE.md`: only `tool_choice="auto"` works here; enforce a required tool step in the
  graph; give thinking-on tool calls a timeout; strip MDX snippet imports.

## Design decisions
- The guard reuses episode 2's router instead of a keyword rule: the router was 16/16 with `json_schema`.
- One reminder only: if the model still answers without searching, the answer is kept (no loop).
- The reminder is a HumanMessage, not a second system prompt, so the transcript shows when the graph intervened.
- The live tests do not use thinking on: a non-stopping call would make the test run for minutes.
- `create_agent` is kept as the comparison. With the same model, tool and prompt it behaved like the hand-written
  graph (2 of 24 vs 2 of 24 searches). The guard needs the graph's own nodes; this episode did not try
  `create_agent` middleware.

## Measurements
- Tool use without a guard, thinking off (docs questions searched, 8 per measurement):

  | variant | before MDX fix | after fix, no final node | final | total |
  | --- | --- | --- | --- | --- |
  | StateGraph, tool description | 2/8 | 0/8 | 0/8 | 2/24 |
  | create_agent, tool description | 1/8 | 0/8 | 1/8 | 2/24 |
  | StateGraph, "Search the docs." | 0/8 | 0/8 | 0/8 | 0/24 |

  Direct questions (8 per measurement) were never searched in any variant (0/96 including thinking on and the guard).
- Guard (StateGraph + router + reminder), thinking off: 8/8 and 8/8 docs questions searched; 8 and 7 of them after
  the reminder (once the model searched on its own); mean 2.12 and 2.06 model calls; mean 11.6 s and 5.25 s per
  question (the GPU had other jobs during the first run). Direct questions: 0/8 searched, no reminder.
- Thinking on, no guard (before the MDX fix): 5 of 8 docs questions searched, 3 of 16 runs timed out at 300 s.
- Queries the model wrote were English keywords: "interrupt resume Command", "checkpointer save state history",
  "StateGraph add_conditional_edges", "create_agent tools parameter". Rank of the expected page in the first search
  (guard, final run): checkpointer 1 and 1, interrupt 1 and 1, conditional edge not in the top 3 (use-subgraphs.mdx,
  streaming.mdx, interrupts.mdx came first), create_agent not in the top 3 (langchain/tools.mdx came first). The raw
  Korean question (episode 2's search, no model) ranks: 1, 1, not in top 3, not in top 3.
- For the conditional-edge question the code that shows `add_conditional_edges` is in snippet components that
  corpus/ does not have, so graph-api-control-flow.mdx lost the word; one guarded answer said the pages do not show
  the exact method.
- Speed (`agent-run.jsonl.txt`): unguarded runs took 1.5-3.0 s (one model call); guarded direct questions 2.9-3.3 s
  (router + one call); the guarded conditional-edge question 9.5 s (5 calls, hit the cap) and 8.0 s (3 calls). The generation rate was
  22-25 tokens/s during the first runaway runs and 46-71 tokens/s later (other GPU processes were running at first).

## Difference from the author's book
- LangGraph book 2-1-2 (page 261613) builds the tool loop the same way as this episode: `@tool`, `llm.bind_tools`,
  `ToolNode`, a routing function `should_call_tools` that checks `last_message.tool_calls`, and `tools → agent`. It
  uses `init_chat_model("openai:gpt-4.1-mini")`; this episode uses `ChatOpenAI` with `base_url` to the local vLLM.
  The book's ReAct graph stops at `MAX_ITERATIONS = 10`; this episode caps at 4 model calls and then answers without
  tools.
- Page 261613 compares `create_agent` and a hand-written `StateGraph` by code size and customization. Measured here
  on the same model: both searched 2 of 24 times; the difference that mattered is that the guard needs nodes of
  one's own.
- LangGraph book 2-2-1 (page 261614) says the LLM decides which tool to call from the tool's name and docstring, and
  that a more specific description gives a more accurate decision. Here: with the specific description 4 of 48
  docs-question runs searched (both builds), with "Search the docs." 0 of 24. The description alone did not make the
  local model search; the graph guard did.
- Page 261614's examples run on `gpt-4.1-mini` (the calculator question calls the tool). The book does not use
  `tool_choice`; on this vLLM only `auto` returns a tool call.
- LangChain book 3-3 (page 261571) writes the weather tool's `Args` line as "날씨를 조회할 도시 이름 (영어)", the same idea
  as this episode's "write the query as English keywords". It also shows `@tool(name=..., description=...,
  return_direct=True)` and error-handling middleware for `create_agent`; this episode uses neither.

## Tests
- Unit: 22 passed (`pytest-unit.txt`): tool schema and description, tool output, the tool loop for both builds, no
  tool call, the cap and final answer, graph shapes, the guard (reminds once, only once, leaves direct questions
  alone), MDX import stripping, plus episode 2's tests.
- Live: 6 passed (`pytest-live.txt`), including the guard making the conditional-edge question search.
