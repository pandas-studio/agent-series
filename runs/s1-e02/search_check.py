"""Rank of the page that answers each docs question: plain word counts vs counts weighted by how rare the word is.

Run from s1-langgraph/: uv run python ../runs/s1-e02/search_check.py
"""
import math

from research_agent.docs import SUFFIXES, WORD, docs_root
from research_agent.route_check import QUESTIONS

EXPECTED = {
    "LangGraph에서 checkpointer는 무엇을 저장하나요?": "langgraph/checkpointers.mdx",
    "interrupt로 멈춘 그래프를 Command(resume=...)로 다시 시작하는 방법은?": "langgraph/interrupts.mdx",
    "StateGraph에서 조건부 엣지는 어떻게 추가하나요?": "langgraph/graph-api-control-flow.mdx",
    "LangChain create_agent에 도구를 넘기는 방법을 알려 주세요.": "langchain/agents.mdx",
}

base = docs_root()
pages = {p.relative_to(base).as_posix(): p.read_text(encoding="utf-8").lower()
         for p in sorted(base.rglob("*")) if p.suffix in SUFFIXES}


def ranking(question: str, weighted: bool) -> list[str]:
    words = {w.lower() for w in WORD.findall(question)}
    df = {w: sum(w in t for t in pages.values()) for w in words}
    weight = {w: math.log((len(pages) + 1) / (df[w] + 1)) if weighted else 1.0 for w in words}
    scores = {path: sum(t.count(w) * weight[w] for w in words) for path, t in pages.items()}
    return [p for p, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) if s > 0]


print(f"{len(pages)} pages")
for question, route in QUESTIONS:
    if route != "docs":
        continue
    want = EXPECTED[question]
    words = sorted({w.lower() for w in WORD.findall(question)})
    df = {w: sum(w in t for t in pages.values()) for w in words}
    plain, weighted = ranking(question, False), ranking(question, True)
    rank = lambda r: r.index(want) + 1 if want in r else None  # noqa: E731
    print(f"{question} | words {df} | expected {want} | plain rank {rank(plain)} (top {plain[0]}) | "
          f"weighted rank {rank(weighted)} (top {weighted[0] if weighted else None})")
