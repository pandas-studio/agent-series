"""Memory check: a multi-turn conversation on one thread_id, turn by turn (run from s1-langgraph/).

Runs the guarded agent with a checkpointer over TURNS on one thread and records, per turn: the route the graph used,
the searches and reminders that happened in this turn, whether the model-call cap ended the turn, the per-turn
fields left in the saved state, the checkpoints written, the input tokens of the turn's first model call and the
start of the answer. A second thread asks the follow-up cold, to show that threads do not share memory.

Run: uv run python -m research_agent.memory_check > ../runs/s1-e04/memory-check.jsonl.txt
"""

import json
import sys
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from .agent import FINAL, REMINDER, build_graph

TURNS = [
    "LangGraph에서 checkpointer는 무엇을 저장하나요?",                         # docs
    "그걸 나중에 다시 꺼내려면 invoke 할 때 무엇을 넘겨야 하나요?",             # docs, needs turn 1
    "HTTP 상태 코드 404는 무슨 뜻인가요?",                                    # direct
    "interrupt로 멈춘 그래프를 Command(resume=...)로 다시 시작하는 방법은?",   # docs
    "지금까지 내가 무엇을 물었는지 한 문장으로 말해 주세요.",                   # direct, needs turns 1-4
]


def turn_row(graph, cfg, question: str, before: int, n_checkpoints: int) -> dict:
    start = time.perf_counter()
    out = graph.invoke({"messages": [{"role": "user", "content": question}]}, cfg)
    seconds = round(time.perf_counter() - start, 2)
    new = out["messages"][before:]
    ai = [m for m in new if isinstance(m, AIMessage)]
    usage = ai[0].usage_metadata if ai and ai[0].usage_metadata else {}
    state = graph.get_state(cfg).values
    return {"question": question, "route": state.get("route"),
            "searches": [c["args"].get("query") for m in ai for c in m.tool_calls],
            "reminded_this_turn": any(isinstance(m, HumanMessage) and m.content == REMINDER for m in new),
            "capped_this_turn": any(isinstance(m, HumanMessage) and m.content == FINAL for m in new),
            "model_calls_this_turn": len(ai),
            "state_fields": sorted(k for k in state if k != "messages"),
            "messages_in_state": len(out["messages"]),
            "checkpoints_written": len(list(graph.get_state_history(cfg))) - n_checkpoints,
            "first_call_input_tokens": usage.get("input_tokens"), "seconds": seconds,
            "answer": out["messages"][-1].content[:160]}


def main() -> int:
    graph = build_graph(guard=True, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "check-a"}}
    for i, q in enumerate(TURNS, 1):
        state = graph.get_state(cfg)
        before = len(state.values.get("messages", [])) if state.values else 0
        n = len(list(graph.get_state_history(cfg)))
        print(json.dumps({"thread": "check-a", "turn": i, **turn_row(graph, cfg, q, before, n)}, ensure_ascii=False),
              flush=True)
    cold = {"configurable": {"thread_id": "check-b"}}
    print(json.dumps({"thread": "check-b", "turn": 1, **turn_row(graph, cold, TURNS[1], 0, 0)}, ensure_ascii=False),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
