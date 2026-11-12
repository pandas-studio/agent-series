# s1-e04 — memory: checkpointer, thread_id, resuming a conversation (2026-11-12)

Facts recorded while building this episode. This is source material for the blog pipeline, not post text.

## Key finding
Adding a checkpointer broke episode 3's guard from the second turn on, because the per-turn state fields carried over
to the next turn: `model_calls` went 3 → 5 → 6 → 7 → 8 over 5 turns, so turn 2 hit the 4-call cap after one search;
`reminded` stayed true, so the docs question in turn 4 was answered without a search or a reminder; and the router
read the thread's first message, so the HTTP 404 question in turn 3 was routed as "docs" (same in 2 runs,
`memory-check-before-r*.jsonl.txt`). Counting those values from the messages since the last user question fixed all
three in both runs (`memory-check-r*.jsonl.txt`).

## What was built
- `src/research_agent/agent.py`:
  - `build_graph(..., checkpointer=None)` passes the checkpointer to `compile()`.
  - `this_turn(messages)` returns the current question (the last user message that is not the graph's own reminder or
    final prompt) and the messages since it.
  - The state keeps only `messages` and `route`. The per-turn values are computed from `this_turn`: model calls this
    turn (for the cap), whether this turn has a search result, whether this turn got the reminder; `classify` routes
    this turn's question. Episode 3's `model_calls` and `reminded` state fields are gone.
- `src/research_agent/chat.py`: a CLI that keeps a thread in a SQLite file across processes
  (`SqliteSaver.from_conn_string`, file `.state/chat.sqlite`, gitignored): `--thread <id> "질문"`, `--history`.
- `src/research_agent/memory_check.py`: 5 turns on one thread (docs, docs follow-up, direct, docs, direct summary) and
  the follow-up alone on a second thread, recording per turn: route, searches, reminder, cap, model calls, messages in
  the thread, checkpoints written, input tokens of the turn's first model call, time.
- `runs/s1-e04/sqlite_check.py`: SqliteSaver with a scripted model (no model server): the connection forms, threads,
  and a second process continuing the thread.
- Packages added directly: `uv add langgraph-checkpoint-sqlite` (3.1.3; it also installed sqlite-vec 0.1.9).
  `InMemorySaver` comes with `langgraph-checkpoint` 4.3.1, already installed with langgraph.
- Installed versions (`versions.txt`): langgraph 1.2.20, langgraph-checkpoint 4.3.1, langgraph-checkpoint-sqlite 3.1.3,
  langchain-core 1.7.6, langchain-openai 1.7.4; vLLM 0.21.0.

## Commands run
- `uv add langgraph-checkpoint-sqlite`
- `uv run python -m research_agent.memory_check > ../runs/s1-e04/memory-check-before-r1.jsonl.txt` (and r2), before
  the fix; the same into `memory-check-r1/r2.jsonl.txt` after it
- `uv run python ../runs/s1-e04/sqlite_check.py` (output `sqlite-check.txt`)
- `uv run python -m research_agent.chat --thread demo "…"` twice and `--history` (`chat-run.txt`)
- `uv run pytest -q` and `AGENT_LIVE=1 uv run pytest -q -m live`

## How Claude Code was used
- Claude Code read the current LangGraph docs first (checkpointers, persistence): `InMemorySaver` ships with langgraph;
  `SqliteSaver` needs `langgraph-checkpoint-sqlite`; the docs' examples open SQLite as
  `SqliteSaver(sqlite3.connect("checkpoint.db"))`.
- Before writing the multi-turn check, Claude Code noticed that episode 3's state had per-turn fields (`model_calls`,
  `reminded`) and that `classify` read `state["messages"][0]`. A 3-turn try on one thread showed `model_calls` at 3, 5,
  6 and `reminded` true after every turn. It then recorded 5 turns twice with that code (`memory-check-before-*`)
  before changing anything.
- The fix counts per-turn values from the messages instead of resetting fields: a graph without a router has no node
  that runs first in every turn, and counted values cannot go stale. "Searched yet" also had to be per turn: it had
  looked at the whole thread, so after turn 1 the guard saw an old search result and never reminded.
- SQLite: Claude Code first wrote the check the way the docs write the connection. `SqliteSaver(sqlite3.connect(path))`
  failed on the first `invoke`, even in the thread that opened the connection: "ProgrammingError: SQLite objects created
  in a thread can only be used in that same thread". The traceback ends in `langgraph/pregel/_loop.py:
  _checkpointer_put_after_previous` running in a `concurrent.futures` worker thread → `SqliteSaver.put`. It failed with
  `durability="sync"`, `"async"` and `"exit"` alike. `sqlite3.connect(path, check_same_thread=False)` and
  `SqliteSaver.from_conn_string(path)` (which opens with `check_same_thread=False`) worked.
- Rules went into `s1-langgraph/CLAUDE.md`: per-turn values come from `this_turn`, not state fields; SQLite through
  `from_conn_string`.

## Design decisions
- Thread ids are given by the caller (`--thread demo`); one thread is one conversation.
- The SQLite file lives in `.state/` and is not committed; `AGENT_CHAT_DB` picks another file.
- `memory_check` uses `InMemorySaver` (one process); persistence across processes is checked separately with SQLite and
  a scripted model, so it does not depend on model output.

## Measurements
- Before the fix (`memory-check-before-r1/r2.jsonl.txt`, 5 turns on thread check-a; same pattern in both runs):
  - turn 1 (docs): searched after the reminder, 3 model calls; state model_calls 3, reminded true.
  - turn 2 (docs follow-up): one search, then the cap ended the turn (final prompt); state model_calls 5.
  - turn 3 (HTTP 404): route "docs" (from the thread's first question), no search; model_calls 6.
  - turn 4 (interrupt/resume, a docs question): no search and no reminder, answered from memory; model_calls 7.
  - turn 5 (summary of the questions): answered from the history; model_calls 8.
  - Checkpoints written per turn: 8, 6, 4, 4, 4.
- After the fix (`memory-check-r1/r2.jsonl.txt`, both runs): turn 1 searched after the reminder (3 calls); turn 2
  searched on its own without a reminder and was not capped (2 calls); turn 3 routed "direct", no search (1 call);
  turn 4 searched on its own (2 calls); turn 5 "direct", answered all four questions from the history (1 call). In a
  thread that already held a search, the model searched by itself in turns 2 and 4 (4 of 4 over 2 runs); alone, the
  same agent searched 2 of 24 times in episode 3.
- Input tokens of each turn's first model call grow with the thread (after the fix, r1 / r2): 394 / 394, 1,556 / 1,641,
  2,664 / 2,766, 2,767 / 2,880, 3,958 / 4,332. Every turn sends the whole history, search results included.
- Checkpoints written per turn after the fix: 8, 6, 4, 6, 4 (8 = input, classify, agent, remind, agent, tools, agent
  and the end of the run; a direct answer writes 4).
- Thread isolation: the follow-up question alone on thread check-b did not know what "그걸" referred to; it searched
  for generic terms ("invoke", "state", "resume execution from previous state").
- SQLite (`sqlite-check.txt`): `connect(path)` failed in the creating thread and from another thread;
  `check_same_thread=False` worked in both (2 and 4 messages); a second process using `from_conn_string` on the same
  file continued the thread (messages 2 → 4; tables checkpoints 6, writes 6, file 28,672 bytes).
- CLI (`chat-run.txt`, real model): two turns in two processes on thread demo; the second answer used the first
  ("그걸" = the checkpoint); `--history` showed 2 questions, 10 messages, 14 checkpoints; the SQLite file was 143,360
  bytes.

## Difference from the author's book
- State carried over between runs: LangGraph book 1-4-1 (page 261599) shows it as the feature that makes memory work:
  in the calculator example the second `invoke({})` keeps `total` and `history` ("병합 결과 … ← 이전 값 유지!"; TIP "새
  입력에 없는 값 → 이전 상태 값 유지"). The same rule broke this episode's agent: fields meant for one turn
  (`model_calls`, `reminded`) were kept for the next turn. The book does not separate per-thread values from per-turn
  values.
- Checkpointer choice: LangGraph book 1-4-2 (page 261600) compares InMemorySaver, SqliteSaver and PostgresSaver and
  recommends SqliteSaver for CLI scripts and single servers. It does not mention the separate package
  (`langgraph-checkpoint-sqlite`) or the thread rule: with langgraph 1.2.20, `SqliteSaver(sqlite3.connect(path))`
  fails on the first invoke; `from_conn_string` or `check_same_thread=False` is needed.
- Memory example: page 261599's chatbot keeps messages with `Annotated[list, add_messages]`; this episode uses
  `MessagesState` (the same reducer) and measures what that costs: the first model call of turn 5 sends about 4,000
  tokens against 394 in turn 1.
- `get_state_history`: LangGraph book 1-4-3 (page 319029) says it returns the newest checkpoint first; this episode
  only counted the checkpoints per turn with it (8, 6, 4, 6, 4) and did not use time travel.

## Tests
- Unit: 27 passed (`pytest-unit.txt`): `this_turn`; a second turn on a thread gets its own reminder, cap budget and
  route; threads do not share messages; `sqlite3.connect(path)` fails on invoke; a thread survives a new SQLite saver;
  plus episodes 2-3.
- Live: 6 passed (`pytest-live.txt`).
