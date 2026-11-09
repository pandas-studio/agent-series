"""The smallest LangGraph program: one node that asks the model.

Run: uv run python -m research_agent.hello_graph "질문"
Episode 2 replaces this with a real state, several nodes and a branch.
"""

import sys

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from .model import make_model


def build(model=None):
    llm = model or make_model()

    def answer(state: MessagesState) -> dict[str, list[AnyMessage]]:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def main() -> int:
    question = " ".join(sys.argv[1:]) or "LangGraph를 한 문장으로 소개해 주세요."
    result = build().invoke({"messages": [("user", question)]})
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
