# s1-e02 — StateGraph basics: state, nodes, edges, a conditional branch (2026-11-10)

Facts recorded while building this episode. This is source material for the blog pipeline, not post text.

## Key finding
The branch is decided by the model, so the router's structured output has to work on the local server. Of the three
`with_structured_output` methods, only `json_schema` worked: 16 of 16 calls valid and correct (8 labelled questions,
2 runs), 0.68–1.1 s per call. `function_calling` gave 0 of 16 (no tool call, empty content) and `json_mode` 0 of 16
(JSON with keys the schema does not have: `answer`, `category`, `decision`). The cause of the `function_calling`
failure is the server, not LangChain: vLLM 0.21 with Qwen returned a tool call only for `tool_choice="auto"`; for
`"required"` (43 tokens generated) and for a named tool, which is what LangChain sends (40 tokens), it returned
none of the generated tokens.

## What was built
- `src/research_agent/graph.py` replaces episode 1's `hello_graph.py`:
  `START → classify → (route == "docs") retrieve → answer → END`, or `classify → answer` when route is "direct".
  - `State` (TypedDict): `question`, `route`, `hits`, `answer`, and `steps: Annotated[list[str], operator.add]`.
    Each node returns only its own name in `steps` and the reducer appends it; the other fields are replaced.
  - `classify`: the model returns a `Route` (Pydantic: `route` is `Literal["docs", "direct"]`, plus `reason`) via
    `with_structured_output(Route, method="json_schema")`.
  - `after_classify` returns `"retrieve"` or `"answer"` (return type `Literal["retrieve", "answer"]`);
    `add_conditional_edges("classify", after_classify, ["retrieve", "answer"])`.
  - `retrieve`: plain Python search over the 16 doc pages in `corpus/` (`src/research_agent/docs.py`), no model.
  - `answer`: the model answers in Korean, from the found excerpts (naming their paths) or without them.
- `src/research_agent/route_check.py`: the classify step with each structured-output method on 8 labelled questions
  (4 docs, 4 direct), twice.
- No new packages: `pydantic` was already installed as a dependency of `langchain-core`.
- Installed versions (`versions.txt`): langgraph 1.2.20, langchain-core 1.7.6, langchain-openai 1.7.4, openai 3.46.0,
  pydantic 2.14.3; vLLM 0.21.0 started with `--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3`.

## Commands run
- `uv run python -m research_agent.route_check > ../runs/s1-e02/route-check.jsonl.txt`
- `uv run python ../runs/s1-e02/tool_choice_check.py` (output `tool-choice-check.txt`)
- `uv run python ../runs/s1-e02/search_check.py` (output `search-check.txt`)
- `uv run pytest -q` and `AGENT_LIVE=1 uv run pytest -q -m live`
- `uv run python -m research_agent.graph "LangGraph에서 checkpointer는 무엇을 저장하나요?"` and
  `uv run python -m research_agent.graph "HTTP 상태 코드 404는 무슨 뜻인가요?"`, twice each (`graph-run.txt`)

## How Claude Code was used
- Before writing the router, Claude Code tried `with_structured_output` with all three methods on one question:
  `json_schema` returned `route='docs'`, `function_calling` returned nothing and no error, `json_mode` returned
  `route` values outside the Literal ("Technical Support / Software Development"). It then measured the three on
  8 labelled questions, twice (`route_check.py`), and kept `json_schema`.
- To find why `function_calling` returned nothing, Claude Code sent the Route tool straight to vLLM with each
  `tool_choice`, thinking off and on (`tool_choice_check.py`), and read what LangChain sends:
  `tool_choice={"type": "function", "function": {"name": "Route"}}`. Only `auto` came back with a tool call.
- The first unit test run failed: the search found nothing for "checkpointer는 무엇을 저장하나요?". The word pattern
  used `\w`, which also matches Hangul, so the Korean particle stuck to the term ("checkpointer는"). Fixed by matching
  ASCII letters, digits, `_` and `-` only.
- The first live graph run picked the wrong pages for the checkpointer question (graph-api-control-flow.mdx,
  graph-api.mdx, graph-api-state.mdx), and the answer said the excerpts did not say what a checkpointer stores
  (`graph-run-before.txt`). The live test still passed, because it only checked that some page path was in the
  answer. Cause: "langgraph" is on all 16 pages and 229 times in graph-api-control-flow.mdx, and the plain word
  counts were dominated by it. Fix: each word's count is weighted by log((pages + 1) / (pages with the word + 1)), so
  a word on every page counts nothing. The live test now checks that checkpointers.mdx comes first and is named in
  the answer.
- A rule went into CLAUDE.md: structured output on the local vLLM uses `method="json_schema"`.

## Design decisions
- The route is a model decision made with structured output, not a keyword rule, because the next episodes route
  between tools and the model's reasons are visible in `Route.reason`.
- The search stays plain Python in this episode; episode 3 turns it into a tool the model calls.
- Unit tests replace the router with a `RunnableLambda` that returns a fixed `Route`, because `GenericFakeChatModel`
  has no structured output; the graph's edges, reducer and search are tested offline.

## Measurements
- Router (`route-check.jsonl.txt`, 8 questions × 2 runs per method): json_schema 16/16 valid, 16/16 correct,
  0.68–1.1 s per call (mean 0.82 s); function_calling 0/16 valid (mean 0.71 s); json_mode 0/16 valid (mean 0.78 s).
- tool_choice (`tool-choice-check.txt`, one question): thinking off — auto: tool call, 61 tokens; required: no tool
  call, 43 tokens; named: no tool call, 40 tokens, finish "stop". Thinking on — auto: tool call; required and named:
  no tool call, only reasoning text.
- Search (`search-check.txt`, rank of the page that answers each docs question, plain counts → weighted counts):
  checkpointer 4 → 1, interrupt/resume 1 → 1, StateGraph conditional edge 4 → 4, create_agent 6 → 4. Korean words
  in the questions ("조건부 엣지", "도구") cannot match the English pages, so two questions still miss.
- Whole graph (`graph-run.txt`, 2 runs): the checkpointer question went classify → retrieve → answer with
  checkpointers.mdx first, 3.56 s and 5.88 s; the 404 question went classify → answer, 4.18 s and 4.28 s.

## Difference from the author's book
LangGraph book 1-3 (pages 261590 basic parts, 261591 state, 261580 nodes, 262302 edges).
- Conditional edges: page 262302 routes with a function on a state value (`if state["score"] > 0.8`) and maps its
  return strings to nodes with a dict (`{"high_quality": "publish", "low_quality": "revise"}`). This episode lets
  the model decide (structured `Route`), and the routing function returns node names typed as
  `Literal["retrieve", "answer"]`, with the list `["retrieve", "answer"]` instead of a dict.
- Reducers: page 261591 shows `messages: Annotated[list, add_messages]` for chat messages. This episode has no
  messages field; it puts `operator.add` on a plain list (`steps`) to record the path through the graph.
- State: the book's first examples (page 261590) use a counter state (`count: int`) and nodes that return the new
  value; same pattern here, with string fields.

## Tests
- Unit: 9 passed, 5 skipped (`pytest-unit.txt`). Live: 5 passed (`pytest-live.txt`).
