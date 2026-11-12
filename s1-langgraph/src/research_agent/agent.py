"""Episode 3: a tool-calling agent, built twice.

build_graph: the loop written by hand with StateGraph, as in the LangGraph quickstart:

    START → agent ─(last message has tool_calls)→ tools → agent …
                  └(no tool_calls)──────────────→ END
    (after MAX_MODEL_CALLS model calls: tools → final → END, a last call without tools)

build_prebuilt: the same model, tool and system prompt through langchain.agents.create_agent.

build_graph(guard=True) adds episode 2's router in front and one check after the agent: a docs question answered
without any search gets one reminder and goes back to the agent. On the local vLLM a forced tool_choice
("required" or a named tool) returned no tool call at all (../runs/s1-e03/force-tool-check.txt), so the graph has
to enforce it.

    START → classify → agent ─(tool_calls)→ tools → agent …
                             ├(docs question, no search yet, no reminder yet)→ remind → agent
                             └(otherwise)→ END

Run: uv run python -m research_agent.agent "질문"            (StateGraph)
     uv run python -m research_agent.agent --guard "질문"    (StateGraph with router and reminder)
     uv run python -m research_agent.agent --prebuilt "질문" (create_agent)
"""

import json
import sys
import time
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .model import make_model
from .tools import search_docs

SYSTEM = """You answer questions for a developer.
If the question is about LangGraph or LangChain, call search_docs first and answer only from the pages it returns,
naming the page paths you used. For any other question, answer directly without tools.
Answer in Korean, in 3-5 sentences."""

REMINDER = "Search the documentation with search_docs before answering, then answer from the pages it returns."
FINAL = "Do not search again. Answer now from the pages above, naming their paths."

# After this many model calls the agent gets no more tools: its pending tool calls run, then one last call without
# tools writes the answer. Before this, a run that hit the cap ended on a tool call and returned an empty answer
# (4 of 128 runs in ../runs/s1-e03/tool-check-before.jsonl.txt and tool-check-nofinal.jsonl.txt).
MAX_MODEL_CALLS = 4


class AgentState(MessagesState):
    route: str


def this_turn(messages: list) -> tuple[str, list]:
    """(question, messages since it) for the current turn: the last user message that is not the graph's own reminder
    or final prompt. With a checkpointer the thread keeps every earlier turn, so per-turn facts (model calls, the
    reminder, whether this turn searched) are counted here instead of being kept in state fields: episode 4 found that
    fields like model_calls and reminded carried over to the next turn (../runs/s1-e04/memory-check-before-r*.jsonl.txt)."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and m.content not in (REMINDER, FINAL):
            return m.content, messages[i:]
    return "", messages


def build_graph(model=None, tools=None, guard: bool = False, router=None, checkpointer=None):
    tools = tools or [search_docs]
    base = model or make_model()
    llm = base.bind_tools(tools)

    def classify(state: AgentState) -> dict:
        from .graph import CLASSIFY_PROMPT, make_router   # episode 2's structured-output router
        question, _ = this_turn(state["messages"])
        return {"route": (router or make_router(base)).invoke(CLASSIFY_PROMPT.format(question=question)).route}

    def agent(state: AgentState) -> dict:
        return {"messages": [llm.invoke([SystemMessage(SYSTEM)] + state["messages"])]}

    def remind(state: AgentState) -> dict:
        return {"messages": [HumanMessage(REMINDER)]}

    def final(state: AgentState) -> dict:
        ask = HumanMessage(FINAL)   # kept in the history, so a run shows that it hit the cap
        reply = base.invoke([SystemMessage(SYSTEM)] + state["messages"] + [ask])
        return {"messages": [ask, reply]}

    def calls(turn: list) -> int:
        return sum(isinstance(m, AIMessage) for m in turn)

    def after_agent(state: AgentState) -> Literal["tools", "remind", "__end__"]:
        _, turn = this_turn(state["messages"])
        if getattr(turn[-1], "tool_calls", None):
            return "tools"
        searched = any(isinstance(m, ToolMessage) for m in turn)
        reminded = any(isinstance(m, HumanMessage) and m.content == REMINDER for m in turn)
        if guard and state.get("route") == "docs" and not searched and not reminded:
            return "remind"
        return END

    def after_tools(state: AgentState) -> Literal["agent", "final"]:
        return "final" if calls(this_turn(state["messages"])[1]) >= MAX_MODEL_CALLS else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("final", final)
    if guard:
        graph.add_node("classify", classify)
        graph.add_node("remind", remind)
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "agent")
        graph.add_edge("remind", "agent")
        graph.add_conditional_edges("agent", after_agent, ["tools", "remind", END])
    else:
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", "final"])
    graph.add_edge("final", END)
    return graph.compile(checkpointer=checkpointer)


def build_prebuilt(model=None, tools=None):
    from langchain.agents import create_agent   # langchain is only needed for this variant

    return create_agent(model or make_model(), tools or [search_docs], system_prompt=SYSTEM)


def summary(messages: list) -> dict:
    """What a run did: the tool calls with their queries, the pages they returned, model calls and the answer."""
    calls = [c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    reminded = any(isinstance(m, HumanMessage) and m.content == REMINDER for m in messages)
    capped = any(isinstance(m, HumanMessage) and m.content == FINAL for m in messages)
    results = [m.content for m in messages if isinstance(m, ToolMessage)]
    pages = [[line[1:line.index("]")] for line in r.splitlines() if line.startswith("[") and "]" in line]
             for r in results]
    usage = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    return {"queries": [c["args"].get("query") for c in calls], "pages": pages,
            "model_calls": sum(isinstance(m, AIMessage) for m in messages),
            "input_tokens": sum(u["input_tokens"] for u in usage), "output_tokens": sum(u["output_tokens"] for u in usage),
            "reminded": reminded, "capped": capped, "answer": messages[-1].content}


def main() -> int:
    args = sys.argv[1:]
    prebuilt, guard = "--prebuilt" in args, "--guard" in args
    question = " ".join(a for a in args if a not in ("--prebuilt", "--guard")) or "StateGraph에서 조건부 엣지는 어떻게 추가하나요?"
    agent = build_prebuilt() if prebuilt else build_graph(guard=guard)
    start = time.perf_counter()
    out = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(json.dumps({"question": question,
                      "build": "create_agent" if prebuilt else "StateGraph + guard" if guard else "StateGraph",
                      "seconds": round(time.perf_counter() - start, 2), **summary(out["messages"])},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
