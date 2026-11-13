"""Where should interrupt() go? A review node of its own vs a pause inside the agent node, on the local Qwen.

Both variants run episode 3's guarded agent (router + one reminder) with a checkpointer and approve every pause:
- "node": build_graph(guard=True, review=True): the agent node calls the model, the review node calls interrupt().
- "in-agent": build_graph(guard=True) with a model wrapper that calls interrupt() right after a model reply with tool
  calls, i.e. inside the agent node. On resume LangGraph re-runs that node from its first line, so the model is called
  again and the call that runs is the new reply, not the one the reviewer saw.

Per run: the calls shown at each pause, the calls that ran (the AI messages that ToolMessages answer), how many
pauses showed a call that then did not run, the agent's model calls (including re-runs) and seconds.
Then a reject pass on the "node" variant: after a refused search, does the model search again (another pause) and does
its answer say that it did not search?

Run from s1-langgraph: uv run python ../runs/s1-e05/review_check.py > ../runs/s1-e05/review-check-r1.jsonl.txt
"""

import json
import sys
import time

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from research_agent.agent import build_graph, pending_calls, this_turn
from research_agent.model import make_model

QUESTIONS = [   # s1-e03 tool_check docs questions
    "LangGraph에서 checkpointer는 무엇을 저장하나요?",
    "interrupt로 멈춘 그래프를 Command(resume=...)로 다시 시작하는 방법은?",
    "StateGraph에서 조건부 엣지는 어떻게 추가하나요?",
    "LangChain create_agent에 도구를 넘기는 방법을 알려 주세요.",
]
REPEATS = 3
MAX_RESUMES = 8


class Counted:
    """Wraps the chat model: counts the agent's tool-bound calls; with pause=True it also calls interrupt() inside the
    agent node right after a reply with tool calls (the 'in-agent' placement)."""

    def __init__(self, real, pause: bool):
        self.real, self.pause, self.calls = real, pause, 0

    def with_structured_output(self, *a, **k):   # episode 2's router
        return self.real.with_structured_output(*a, **k)

    def invoke(self, *a, **k):                    # the 'final' node (no tools)
        return self.real.invoke(*a, **k)

    def bind_tools(self, tools, **k):
        bound, outer = self.real.bind_tools(tools, **k), self

        class Bound:
            def invoke(self, messages, *a, **kw):
                outer.calls += 1
                reply = bound.invoke(messages, *a, **kw)
                if outer.pause and reply.tool_calls:
                    interrupt({"question": "Run these searches?", "tool_calls": pending_calls(reply)})
                return reply

        return Bound()


def key(calls: list[dict]) -> list:
    return [(c["name"], json.dumps(c["args"], sort_keys=True, ensure_ascii=False)) for c in calls]


def run(variant: str, question: str, thread: str, decide=lambda i: {"action": "approve"}) -> dict:
    model = Counted(make_model(), pause=(variant == "in-agent"))
    graph = build_graph(model, guard=True, review=(variant == "node"), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": thread}}
    t0 = time.perf_counter()
    shown = []
    try:
        out = graph.invoke({"messages": [{"role": "user", "content": question}]}, cfg)
        while "__interrupt__" in out and len(shown) < MAX_RESUMES:
            shown.append(out["__interrupt__"][0].value["tool_calls"])
            out = graph.invoke(Command(resume=decide(len(shown))), cfg)
    except Exception as e:  # noqa: BLE001 - a failed run is a measurement too
        return {"variant": variant, "question": question, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "seconds": round(time.perf_counter() - t0, 2)}
    _, turn = this_turn(out["messages"])
    answered = {m.tool_call_id for m in turn if isinstance(m, ToolMessage)}
    ran = [pending_calls(m) for m in turn if isinstance(m, AIMessage) and m.tool_calls
           and all(c["id"] in answered for c in m.tool_calls)]
    ran_keys = [key(r) for r in ran]
    mismatch = sum(1 for s in shown if key(s) not in ran_keys)
    answer = out["messages"][-1].content if isinstance(out["messages"][-1], AIMessage) else ""
    return {"variant": variant, "question": question, "pauses": len(shown),
            "shown": [[c["args"].get("query") for c in s] for s in shown],
            "ran": [[c["args"].get("query") for c in r] for r in ran],
            "shown_not_run": mismatch, "agent_model_calls": model.calls,
            "still_paused": "__interrupt__" in out, "answer_chars": len(answer),
            "says_no_search": any(w in answer for w in ("검색하지", "검색 없이", "검색을 하지", "검색을 수행하지")),
            "seconds": round(time.perf_counter() - t0, 2), "answer": answer[:160]}


def main(tag: str) -> int:
    rows = []
    for variant in ("node", "in-agent"):
        for r in range(REPEATS):
            for qi, q in enumerate(QUESTIONS):
                row = run(variant, q, f"{tag}-{variant}-{r}-{qi}")
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    for variant in ("node", "in-agent"):
        mine = [x for x in rows if x["variant"] == variant and "error" not in x]
        print(json.dumps({"summary": variant, "runs": len(mine), "pauses": sum(x["pauses"] for x in mine),
                          "shown_not_run": sum(x["shown_not_run"] for x in mine),
                          "runs_with_mismatch": sum(1 for x in mine if x["shown_not_run"]),
                          "agent_model_calls": sum(x["agent_model_calls"] for x in mine),
                          "mean_seconds": round(sum(x["seconds"] for x in mine) / max(1, len(mine)), 2),
                          "errors": sum(1 for x in rows if x["variant"] == variant and "error" in x)},
                         ensure_ascii=False), flush=True)
    # Reject pass: refuse the first search, approve any later one.
    rej = []
    for r in range(2):
        for qi, q in enumerate(QUESTIONS):
            row = run("node", q, f"{tag}-reject-{r}-{qi}",
                      decide=lambda i: {"action": "reject", "reason": "검색 없이 답해 주세요"} if i == 1
                      else {"action": "approve"})
            row["variant"] = "node-reject-first"
            rej.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    ok = [x for x in rej if "error" not in x]
    print(json.dumps({"summary": "node-reject-first", "runs": len(ok),
                      "searched_again_after_reject": sum(1 for x in ok if x["pauses"] > 1),
                      "says_no_search": sum(x["says_no_search"] for x in ok)},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "r1"))
