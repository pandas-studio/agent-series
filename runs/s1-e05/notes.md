# s1-e05 — human review before each search: interrupt() and Command(resume=...) (2026-11-14)

Facts recorded while building this episode. This is source material for the blog pipeline, not post text.

## Key finding
Where `interrupt()` sits decides whether the reviewer approves the call that runs. With the pause in a `review` node
of its own, the search that ran was the one shown at the pause in every case (0 of 30 pauses differed, 2 runs). With
the pause inside the agent node, right after the model call, LangGraph re-ran the node on resume, called the model
again, and in 27 of 34 pauses the search that ran differed from the one the reviewer approved (20 of 24 runs; one
approved `create_agent tools parameter` and ran `create_react_agent tools`; `review-check-r*.jsonl.txt`). Separately,
the book's first sensitive-tool example (page 261618) still ran `delete_file` after the reviewer rejected it
(`book-check.jsonl.txt`).

## What was built
- `s1-langgraph/src/research_agent/agent.py`: `build_graph(..., review=True)`.
  - Graph: `agent ─(tool_calls)→ review ─(approve / edit)→ tools → agent`, `review ─(reject)→ agent` (or `final` at
    the model-call cap). Without `review=True` the graph is episode 4's.
  - `review_node` calls `interrupt({"question": "Run these searches?", "tool_calls": [...]},
    response_schema=ReviewDecision)` as its first statement and returns `Command(goto=...)`.
  - `ReviewDecision` (pydantic): `action` = `approve` | `edit` | `reject`, `query` (edit), `reason` (reject).
  - approve → `Command(goto="tools")`.
  - edit → `with_query(message, query)`: a copy of the AI message with the same id and the new query, so the messages
    reducer replaces the model's message instead of adding one; then `tools`.
  - reject → one `ToolMessage(status="error")` per tool call id, text `REJECTED` ("The reviewer did not allow this
    search (...). Answer without searching, and say that you did not search."), then `Command(goto="agent")`; the
    tools node is never reached.
- `chat.py` (episode 4's SQLite chat): `--review` pauses before each search; `--approve`, `--edit QUERY`,
  `--reject REASON` resume the paused thread, each in a new process. `waiting_on()` reads `get_state(cfg).next`
  (`("review",)` while paused).
- `tests/test_review.py`: 7 unit tests (pause payload and schema, approve, edit replaces the same message, reject runs
  no search, invalid resume value, no pause without review, resume from a second SQLite saver) and 1 live test.
- `runs/s1-e05/review_check.py` (placements), `book_check.py` (the book's examples on the installed langgraph),
  `chat-run.txt` (the chat commands on the live model).
- No packages added. Versions (`versions.txt`): langgraph 1.2.20, langchain-core 1.7.6, langgraph-checkpoint-sqlite
  3.1.3, pydantic 2.14.3; model Qwen3.6-35B-A3B-NVFP4 on vLLM 0.21.0, thinking off.

## Commands run
- `uv run python ../runs/s1-e05/book_check.py` (`book-check.jsonl.txt`)
- `uv run python ../runs/s1-e05/review_check.py r1` and `... r2` (`review-check-r1.jsonl.txt`, `review-check-r2.jsonl.txt`)
- `uv run python -m research_agent.chat --thread r1 --review "interrupt로 멈춘 그래프는 어떻게 다시 시작하나요?"`, then
  `--thread r1 --edit "interrupt Command resume"`; `--thread r2 --review "LangGraph에서 checkpointer는 무엇을 저장하나요?"`,
  then `--thread r2 --reject "검색 없이 답해 주세요"`; `--thread r1 --history` (`chat-run.txt`)
- `uv run pytest -q` (34 passed, 7 skipped) and `AGENT_LIVE=1 uv run pytest -q -m live` (7 passed)

## How Claude Code was used
- Claude Code read the current LangGraph interrupts page (docs-langchain MCP) first: a checkpointer and a
  `thread_id` are needed; `invoke` returns the pause under `__interrupt__`; "The node restarts from the beginning of
  the node where the interrupt was called when resumed, so any code before the interrupt runs again";
  `response_schema` (langgraph>=1.2.12) validates resume values; the page's tool example puts `interrupt()` inside
  the tool function.
- It read the two book pages (LangGraph book 2-4-1, page 261618; 2-4-2, page 261619) and ran their examples with a
  scripted fake model before writing the agent (`book_check.py`). The first result was unexpected: in the page 261618
  sensitive-tool graph a rejected `delete_file` still ran.
- To compare placements on the live model without adding a wrong variant to the library, `review_check.py` wraps the
  model: the wrapper calls `interrupt()` right after a reply with tool calls, which puts the pause inside the agent
  node of the same `build_graph(guard=True)` graph.
- The first draft of `chat.py`'s waiting check was convoluted; it became `waiting_on()` with `get_state(cfg).next`.
- The new live test first lacked `@pytest.mark.live`, so `-m live` ran 6 tests instead of 7; fixed.
- Rules went into `s1-langgraph/CLAUDE.md`: `interrupt()` in its own node; a rejected call must not reach `ToolNode`;
  no `try/except Exception` around `interrupt()`; resume values validated with `response_schema`.

## Design decisions
- Review every `search_docs` call, though a search is harmless: the episode is about the mechanism, and the edit
  action has a real use here (episode 2 found that Korean words in a query match nothing; the reviewer can replace the
  query with English keywords).
- The pause is a node of its own after the agent node, not inside the tool and not inside the agent node.
  - Inside the tool (the docs' pattern): ToolNode runs parallel tool calls, so two calls in one AI message pause twice
    in one step and the resume needs a map of interrupt ids. Not measured here.
  - Inside the agent node: measured below.
- Reject answers every call id with a ToolMessage and routes past the tools node with `Command(goto="agent")`: the
  next model call needs a reply for each tool call id, and ToolNode would otherwise run the calls.
- An edit sets the same query on every pending call (the local Qwen sent one call per message in these runs, except
  where noted).

## Measurements
### Placement of interrupt() (`review-check-r1.jsonl.txt`, `review-check-r2.jsonl.txt`)
The 4 docs questions of s1-e03, 3 times each, per variant, 2 runs; every pause approved. Numbers are r1 / r2.
- `review` node: 14 / 16 pauses; shown call not run 0 / 0; agent model calls 36 / 36. Per run (both passes): 3 calls in 18
  runs (a reply without a search, the guard's reminder, then the search call and the answer), 2 in 3 runs (search
  call, answer), 4 in 3 runs (3 searches, then the model-call cap and the `final` node); mean 6.08 s / 5.89 s per question.
- Pause inside the agent node: 19 / 15 pauses; shown call not run 15 / 12 (runs with at least one: 11 of 12 / 9 of 12);
  agent model calls 57 / 51; mean 6.52 s / 6.64 s.
- Of the 27 pauses whose call did not run, 1 differed only in letter case (`interrupt resume Command` →
  `interrupt resume command`); the others had other words, e.g. `checkpointer` → `LangGraph checkpointer`,
  `checkpointer` → `checkpointer save state history`, `LangChain create_agent tools parameter` →
  `create_react_agent tools` (a different API name).
- Why: on resume LangGraph re-ran the agent node from its first line; the model was called again (temperature 0.6)
  and returned a new reply, and `interrupt()` returned the resume value at once, so the new reply went on to the tools.
  The extra model calls (57 − 36, 51 − 36) are these re-runs.
- Reject the first search, approve any later one (`node-reject-first`, 8 runs per pass): the model searched again in
  0 of 8 / 0 of 8; its answer said that it had not searched in 1 of 8 / 3 of 8, although the ToolMessage asks it to
  say so.

### The chat across processes (`chat-run.txt`)
- `--review` paused after 3.27 s with `search_docs` query `interrupt resume graph`; `--edit "interrupt Command resume"`
  (a new process) ran the edited query, top page `langgraph/interrupts.mdx`, answer in 2.04 s.
- Thread r2 paused with query `checkpointer stores what`; `--reject` answered without a search (1.51 s).
- `--history` of r1: 6 messages, 9 checkpoints.

### The book's examples on langgraph 1.2.20 (`book-check.jsonl.txt`, scripted fake model)
- Page 261618 sensitive-tool graph, `delete_file` call, resume `"reject"`: tools executed `delete_file(/tmp/important.txt)`;
  messages: the rejection ToolMessage, then "파일 삭제 완료: /tmp/important.txt". With a safe and a sensitive call in one
  message, both ran.
- Page 261619 rule 1, `interrupt()` inside `try/except Exception`: the graph did not pause; the node returned
  "에러 발생 (GraphInterrupt)".
- Page 261619 rule 4: the code before `interrupt()` ran 2 times, the code after it once.
- No checkpointer: `invoke` returned `__interrupt__` without an error; `Command(resume=...)` then raised
  `RuntimeError: Cannot use Command(resume=...) without checkpointer` (with and without a thread_id).

## Difference from the author's book
- Rejected call still runs: page 261618 introduces "특정 도구가 호출될 때만 사람의 승인을 요구하는 패턴입니다." Its
  `check_sensitive_tools` returns a ToolMessage on reject (comment "# 거부 시 도구 호출을 취소하고 메시지로 대체"), and the
  graph has `graph.add_edge("check_sensitive", "tools")`. On langgraph 1.2.20 ToolNode then ran the rejected
  `delete_file` (Measurements). Page 261619's version routes the rejection with `Command(goto="reasoning")`: "거부 시
  `Command(goto="reasoning")`으로 도구 실행을 건너뛰고 reasoning 노드로 되돌아갑니다." This episode does the same
  (`Command(goto="agent")`).
- Where the node restarts: page 261618 says "`Command(resume=...)`로 중단된 지점부터 재개됩니다." Page 261619 rule 4 adds
  "`interrupt()` 이전의 코드는 재개 시 **다시 실행**될 수 있습니다." On langgraph 1.2.20 the code before `interrupt()`
  ran again every time (2 of 2), and in the agent it changed the executed call (27 of 34 pauses).
- Checkpointer: page 261618 says "`interrupt()`를 사용하려면 반드시 Checkpointer가 필요합니다. Checkpointer가 없으면 중단된
  상태를 저장할 수 없어 재개가 불가능합니다." Without one, the graph paused and returned `__interrupt__`; only the resume
  failed (`RuntimeError`).
- Resume value: page 261618 says "`interrupt()`를 직접 사용하는 StateGraph 방식에서는 `Command(resume="approve")` 같이 간단한
  값을 전달합니다." This episode resumes with a dict validated by `response_schema=ReviewDecision` (langgraph>=1.2.12,
  not in the book); an invalid value raised `ValidationError` and the thread stayed paused.
- Rule 1 held: page 261619 says "`interrupt()`는 내부적으로 특수한 예외를 발생시켜 그래프를 중단합니다."; inside
  `try/except Exception` the `GraphInterrupt` was caught and the graph did not pause.

## Tests
- Unit: 34 passed, 7 skipped (`pytest-unit.txt`): episode 4's 27 plus 7 in `test_review.py`.
- Live: 7 passed (`pytest-live.txt`): episode 4's 6 plus `test_live_review_pauses_then_approve_answers`.
