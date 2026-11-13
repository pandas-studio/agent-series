"""Episode 4: talk to the guarded agent on a thread that survives the process, saved in a SQLite file.

Run: uv run python -m research_agent.chat --thread demo "LangGraph에서 checkpointer는 무엇을 저장하나요?"
     uv run python -m research_agent.chat --thread demo "그걸 다시 꺼내려면 무엇을 넘겨야 하나요?"
     uv run python -m research_agent.chat --thread demo --history        (the thread's turns and checkpoints)

Episode 5, review before every search (each command can be a new process; the pause lives in the SQLite file):
     uv run python -m research_agent.chat --thread r1 --review "interrupt로 멈춘 그래프는 어떻게 다시 시작하나요?"
     uv run python -m research_agent.chat --thread r1 --approve
     uv run python -m research_agent.chat --thread r1 --edit "interrupt Command resume"
     uv run python -m research_agent.chat --thread r1 --reject "검색 없이 답해 주세요"

The file is AGENT_CHAT_DB (default .state/chat.sqlite in this folder, not committed).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .agent import FINAL, REMINDER, build_graph, summary, this_turn

DEFAULT_DB = Path(__file__).resolve().parents[2] / ".state" / "chat.sqlite"


def open_saver(path: str | Path):
    """SqliteSaver.from_conn_string opens the file with check_same_thread=False. A plain sqlite3.connect(path) fails
    on the first invoke: LangGraph writes checkpoints from a worker thread (../runs/s1-e04/sqlite-check.txt)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(path))


def waiting_on(saver, cfg: dict) -> tuple:
    """The nodes the thread will run next, from its last checkpoint: ("review",) while a search waits for review."""
    return build_graph(guard=True, checkpointer=saver, review=True).get_state(cfg).next


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--thread", default="default")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--review", action="store_true", help="pause before every search (episode 5)")
    ap.add_argument("--approve", action="store_true", help="resume a paused thread: run the search")
    ap.add_argument("--edit", metavar="QUERY", help="resume a paused thread: search for QUERY instead")
    ap.add_argument("--reject", metavar="REASON", help="resume a paused thread: no search")
    a = ap.parse_args()
    resume = ({"action": "approve"} if a.approve else {"action": "edit", "query": a.edit} if a.edit
              else {"action": "reject", "reason": a.reject} if a.reject else None)
    cfg = {"configurable": {"thread_id": a.thread}}
    with open_saver(os.environ.get("AGENT_CHAT_DB", DEFAULT_DB)) as saver:
        paused = waiting_on(saver, cfg) == ("review",)
        graph = build_graph(guard=True, checkpointer=saver, review=a.review or paused or resume is not None)
        if a.history:
            state = graph.get_state(cfg)
            questions = [m.content for m in state.values.get("messages", [])
                         if isinstance(m, HumanMessage) and m.content not in (REMINDER, FINAL)]
            print(json.dumps({"thread": a.thread, "questions": questions,
                              "messages": len(state.values.get("messages", [])),
                              "checkpoints": len(list(graph.get_state_history(cfg)))}, ensure_ascii=False, indent=2))
            return 0
        start = time.perf_counter()
        if resume is not None:
            if not paused:
                print(json.dumps({"thread": a.thread, "error": "this thread is not waiting for a review"}))
                return 1
            out = graph.invoke(Command(resume=resume), cfg)
        else:
            question = " ".join(a.question) or "LangGraph에서 checkpointer는 무엇을 저장하나요?"
            out = graph.invoke({"messages": [{"role": "user", "content": question}]}, cfg)
        if "__interrupt__" in out:
            print(json.dumps({"thread": a.thread, "seconds": round(time.perf_counter() - start, 2), "paused": True,
                              "pending": out["__interrupt__"][0].value["tool_calls"],
                              "resume_with": "--approve | --edit QUERY | --reject REASON"}, ensure_ascii=False, indent=2))
            return 0
        _, turn = this_turn(out["messages"])
        print(json.dumps({"thread": a.thread, "seconds": round(time.perf_counter() - start, 2),
                          "messages_in_thread": len(out["messages"]), **summary(turn)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
