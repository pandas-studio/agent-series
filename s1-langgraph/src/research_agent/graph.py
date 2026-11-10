"""Episode 2: a graph with a typed state, three nodes and a conditional edge.

    START → classify ─(route == "docs")→ retrieve → answer → END
                     └(route == "direct")────────→ answer

classify asks the model for a structured Route; retrieve searches the local doc pages in plain Python; answer asks
the model with the found pages, or without them.

Run: uv run python -m research_agent.graph "질문"
"""

import json
import operator
import sys
import time
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

from .docs import search
from .model import make_model


class Route(BaseModel):
    """Where a question goes next."""

    route: Literal["docs", "direct"] = Field(
        description="docs: the answer needs the local LangGraph/LangChain documentation; direct: a general question "
        "that can be answered without it")
    reason: str = Field(description="one short sentence")


class State(TypedDict, total=False):
    question: str
    route: str
    hits: list[dict]
    answer: str
    # A reducer: each node returns only its own name and LangGraph appends it. Without the reducer the last
    # node's list would replace the earlier ones.
    steps: Annotated[list[str], operator.add]


CLASSIFY_PROMPT = """Decide where this question should go.
- docs: it asks about LangGraph or LangChain (APIs, configuration, behaviour), so the answer should come from the local
  documentation pages.
- direct: anything else; answer it without the documentation.

Question: {question}"""

ANSWER_PROMPT = """Answer the question in Korean, in 3-5 sentences.
{context}
Question: {question}"""

DOCS_CONTEXT = """Use only these documentation excerpts, and name the page paths you used.

{pages}
"""


def make_router(model, method: str = "json_schema"):
    """Structured output for the route. method="json_schema" sends the schema to the server (vLLM enforces it);
    see route_check.py for how the other methods behave on a local Qwen."""
    return model.with_structured_output(Route, method=method)


def build(model=None, router=None, docs_root=None):
    llm = model or make_model()
    route_model = router or make_router(llm)

    def classify(state: State) -> dict:
        r = route_model.invoke(CLASSIFY_PROMPT.format(question=state["question"]))
        return {"route": r.route, "steps": ["classify"]}

    def retrieve(state: State) -> dict:
        return {"hits": search(state["question"], root=docs_root), "steps": ["retrieve"]}

    def answer(state: State) -> dict:
        hits = state.get("hits") or []
        pages = "\n\n".join(f"[{h['path']}]\n{h['excerpt']}" for h in hits)
        context = DOCS_CONTEXT.format(pages=pages) if hits else ""
        reply = llm.invoke(ANSWER_PROMPT.format(context=context, question=state["question"]))
        return {"answer": reply.content, "steps": ["answer"]}

    def after_classify(state: State) -> Literal["retrieve", "answer"]:
        return "retrieve" if state["route"] == "docs" else "answer"

    graph = StateGraph(State)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", after_classify, ["retrieve", "answer"])
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def main() -> int:
    question = " ".join(sys.argv[1:]) or "LangGraph에서 checkpointer는 무엇을 저장하나요?"
    start = time.perf_counter()
    out = build().invoke({"question": question, "steps": []})
    print(json.dumps({"question": question, "route": out["route"], "steps": out["steps"],
                      "pages": [h["path"] for h in out.get("hits") or []],
                      "seconds": round(time.perf_counter() - start, 2), "answer": out["answer"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
