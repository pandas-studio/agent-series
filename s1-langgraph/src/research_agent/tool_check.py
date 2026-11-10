"""Tool check: does the model call search_docs, with what query, and does the query find the right page?

Episode 2's 8 labelled questions (4 docs, 4 direct), run twice per variant:
- graph / prebuilt: build_graph (StateGraph) or build_prebuilt (create_agent), same model, tool and system prompt;
- doc: the tool's own description, or VAGUE_DOC ("Search the docs.");
- thinking: off (default) or on;
- guard: build_graph(guard=True), the router and one reminder for a docs question answered without a search.

Run: uv run python -m research_agent.tool_check > ../runs/s1-e03/tool-check.jsonl.txt
     uv run python -m research_agent.tool_check --only guard > ../runs/s1-e03/tool-check-guard.jsonl.txt
     TOOL_CHECK_THINKING=0 skips the thinking-on variant (minutes per run when the model does not stop).
One JSON line per run and a summary line per variant at the end.
"""

import json
import os
import sys
import time

from .agent import build_graph, build_prebuilt, summary
from .config import load_settings
from .docs import search
from .model import make_model
from .route_check import QUESTIONS
from .tools import VAGUE_DOC, search_docs, with_description

# The page that answers each docs question (episode 2, search-check.txt).
EXPECTED = {
    "LangGraph에서 checkpointer는 무엇을 저장하나요?": "langgraph/checkpointers.mdx",
    "interrupt로 멈춘 그래프를 Command(resume=...)로 다시 시작하는 방법은?": "langgraph/interrupts.mdx",
    "StateGraph에서 조건부 엣지는 어떻게 추가하나요?": "langgraph/graph-api-control-flow.mdx",
    "LangChain create_agent에 도구를 넘기는 방법을 알려 주세요.": "langchain/agents.mdx",
}
VARIANTS = [("graph", "tool doc", False), ("prebuilt", "tool doc", False), ("graph", "vague", False),
            ("graph", "tool doc", True), ("guard", "tool doc", False)]


def run(build: str, doc: str, thinking: bool, question: str, expected: str) -> dict:
    model = make_model(load_settings({"AGENT_THINKING": "1" if thinking else "0"} | _env()))
    tool = search_docs if doc == "tool doc" else with_description(VAGUE_DOC)
    agent = {"graph": lambda: build_graph(model, [tool]), "prebuilt": lambda: build_prebuilt(model, [tool]),
             "guard": lambda: build_graph(model, [tool], guard=True)}[build]()
    start = time.perf_counter()
    try:
        out = agent.invoke({"messages": [{"role": "user", "content": question}]})
    except Exception as e:   # a request that times out is a result too (thinking on: one did at 120 s)
        return {"build": build, "doc": doc, "thinking": thinking, "question": question, "expected": expected,
                "searched": None, "ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}",
                "seconds": round(time.perf_counter() - start, 2)}
    s = summary(out["messages"])
    want = EXPECTED.get(question)
    first = s["pages"][0] if s["pages"] else []
    return {"build": build, "doc": doc, "thinking": thinking, "question": question, "expected": expected,
            "searched": bool(s["queries"]), "ok": bool(s["queries"]) == (expected == "docs"),
            "queries": s["queries"], "first_pages": first,
            "rank": (first.index(want) + 1 if want in first else None) if want and s["queries"] else None,
            "reminded": s["reminded"], "model_calls": s["model_calls"], "input_tokens": s["input_tokens"], "output_tokens": s["output_tokens"],
            "seconds": round(time.perf_counter() - start, 2), "answer": s["answer"][:160]}


def _env() -> dict:
    return {k: v for k, v in os.environ.items() if k.startswith("AGENT_") and k != "AGENT_THINKING"}


def main(only: str | None = None) -> int:
    """only: run just the variants of this build (e.g. "guard"); the baseline lines are printed either way."""
    # Baseline: episode 2's search with the raw question (no model): rank of the expected page in the top 3.
    for q, want in EXPECTED.items():
        top = [h["path"] for h in search(q, limit=3)]
        print(json.dumps({"baseline": "raw question", "question": q, "top3": top,
                          "rank": top.index(want) + 1 if want in top else None}, ensure_ascii=False), flush=True)
    rows = []
    for rnd in (1, 2):
        for build, doc, thinking in VARIANTS:
            if (only and build != only) or (thinking and os.environ.get("TOOL_CHECK_THINKING") == "0"):
                continue
            for question, expected in QUESTIONS:
                row = dict(run(build, doc, thinking, question, expected), run=rnd)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    summarize(rows)
    return 0


def summarize(rows: list) -> None:
    """One summary line per variant (also: python -m research_agent.tool_check --summary FILE, from saved rows)."""
    for build, doc, thinking in VARIANTS:
        mine = [r for r in rows if (r["build"], r["doc"], r["thinking"]) == (build, doc, thinking)]
        if not mine:
            continue
        docs_q = [r for r in mine if r["expected"] == "docs"]
        direct_q = [r for r in mine if r["expected"] == "direct"]
        print(json.dumps({"summary": f"{build} / {doc} / thinking {'on' if thinking else 'off'}",
                          "docs_searched": f"{sum(bool(r['searched']) for r in docs_q)}/{len(docs_q)}",
                          "direct_searched": f"{sum(bool(r['searched']) for r in direct_q)}/{len(direct_q)}",
                          "rank1": f"{sum(r.get('rank') == 1 for r in docs_q)}/{len(docs_q)}",
                          "in_top3": f"{sum(r.get('rank') is not None for r in docs_q)}/{len(docs_q)}",
                          "reminded": sum(bool(r.get("reminded")) for r in mine),
                          "errors": sum("error" in r for r in mine),
                          "mean_seconds": round(sum(r["seconds"] for r in mine) / len(mine), 2),
                          "mean_model_calls": round(sum(r.get("model_calls", 0) for r in mine) / len(mine), 2)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--summary"]:
        summarize([json.loads(ln) for ln in open(sys.argv[2], encoding="utf-8") if ln.startswith('{"build"')])
        sys.exit(0)
    sys.exit(main(sys.argv[2] if sys.argv[1:2] == ["--only"] else None))
