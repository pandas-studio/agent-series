"""Episode 4: talk to the guarded agent on a thread that survives the process, saved in a SQLite file.

Run: uv run python -m research_agent.chat --thread demo "LangGraph에서 checkpointer는 무엇을 저장하나요?"
     uv run python -m research_agent.chat --thread demo "그걸 다시 꺼내려면 무엇을 넘겨야 하나요?"
     uv run python -m research_agent.chat --thread demo --history        (the thread's turns and checkpoints)

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

from .agent import FINAL, REMINDER, build_graph, summary, this_turn

DEFAULT_DB = Path(__file__).resolve().parents[2] / ".state" / "chat.sqlite"


def open_saver(path: str | Path):
    """SqliteSaver.from_conn_string opens the file with check_same_thread=False. A plain sqlite3.connect(path) fails
    on the first invoke: LangGraph writes checkpoints from a worker thread (../runs/s1-e04/sqlite-check.txt)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--thread", default="default")
    ap.add_argument("--history", action="store_true")
    a = ap.parse_args()
    cfg = {"configurable": {"thread_id": a.thread}}
    with open_saver(os.environ.get("AGENT_CHAT_DB", DEFAULT_DB)) as saver:
        graph = build_graph(guard=True, checkpointer=saver)
        if a.history:
            state = graph.get_state(cfg)
            questions = [m.content for m in state.values.get("messages", [])
                         if isinstance(m, HumanMessage) and m.content not in (REMINDER, FINAL)]
            print(json.dumps({"thread": a.thread, "questions": questions,
                              "messages": len(state.values.get("messages", [])),
                              "checkpoints": len(list(graph.get_state_history(cfg)))}, ensure_ascii=False, indent=2))
            return 0
        question = " ".join(a.question) or "LangGraph에서 checkpointer는 무엇을 저장하나요?"
        start = time.perf_counter()
        out = graph.invoke({"messages": [{"role": "user", "content": question}]}, cfg)
        _, turn = this_turn(out["messages"])
        print(json.dumps({"thread": a.thread, "seconds": round(time.perf_counter() - start, 2),
                          "messages_in_thread": len(out["messages"]), **summary(turn)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
