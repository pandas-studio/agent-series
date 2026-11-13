"""Run the author's book examples (LangGraph book 2-4-1 / 2-4-2, pages 261618 and 261619) on the installed langgraph,
with a scripted fake model instead of gpt-4.1-mini. Prints one JSON line per check.

Run from s1-langgraph: uv run python ../runs/s1-e05/book_check.py > ../runs/s1-e05/book-check.jsonl.txt
"""

import json
from importlib.metadata import version

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

RAN: list[str] = []   # tool bodies append here, so a check can see what actually executed


def out(check: str, **fields) -> None:
    print(json.dumps({"check": check} | fields, ensure_ascii=False, default=str), flush=True)


class Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    RAN.append(f"web_search({query})")
    return f"'{query}'에 대한 검색 결과입니다."


@tool
def delete_file(path: str) -> str:
    """파일을 삭제합니다."""
    RAN.append(f"delete_file({path})")
    return f"파일 삭제 완료: {path}"


SENSITIVE_TOOLS = {"delete_file"}


def calls(*tcs) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"} for n, a, i in tcs])


def book_261618_graph(model):
    """Page 261618 '민감 도구 승인 패턴': check_sensitive_tools → (plain edge) → tools."""
    def reasoning_node(state):
        return {"messages": [model.invoke(state["messages"])]}

    def check_sensitive_tools(state):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return {}
        for tc in last.tool_calls:
            if tc["name"] in SENSITIVE_TOOLS:
                response = interrupt({"tool": tc["name"], "args": tc["args"]})
                if response != "approve":
                    return {"messages": [ToolMessage(content=f"사용자가 '{tc['name']}' 실행을 거부했습니다.",
                                                     tool_call_id=tc["id"])]}
        return {}

    def should_continue(state):
        last = state["messages"][-1]
        return "check_sensitive" if getattr(last, "tool_calls", None) else END

    g = StateGraph(MessagesState)
    g.add_node("reasoning", reasoning_node)
    g.add_node("check_sensitive", check_sensitive_tools)
    g.add_node("tools", ToolNode([web_search, delete_file]))
    g.add_edge(START, "reasoning")
    g.add_conditional_edges("reasoning", should_continue, {"check_sensitive": "check_sensitive", END: END})
    g.add_edge("check_sensitive", "tools")
    g.add_edge("tools", "reasoning")
    return g.compile(checkpointer=InMemorySaver())


def run_261618_reject() -> None:
    RAN.clear()
    model = Fake(messages=iter([calls(("delete_file", {"path": "/tmp/important.txt"}, "d1")), AIMessage("끝")]))
    app = book_261618_graph(model)
    cfg = {"configurable": {"thread_id": "t"}}
    try:
        app.invoke({"messages": [{"role": "user", "content": "중요한 파일을 삭제해 줘"}]}, cfg)
        res = app.invoke(Command(resume="reject"), cfg)
        tool_msgs = [m.content for m in res["messages"] if isinstance(m, ToolMessage)]
        out("261618 reject → plain edge to ToolNode", tools_executed=list(RAN), tool_messages=tool_msgs,
            final=res["messages"][-1].content)
    except Exception as e:  # noqa: BLE001
        out("261618 reject → plain edge to ToolNode", tools_executed=list(RAN),
            error=f"{type(e).__name__}: {str(e)[:300]}")


def run_261618_mixed_reject() -> None:
    """A safe and a sensitive call in one message, sensitive rejected (the page's case: search runs, email waits)."""
    RAN.clear()
    model = Fake(messages=iter([calls(("web_search", {"query": "회의"}, "w1"),
                                      ("delete_file", {"path": "/tmp/important.txt"}, "d1")), AIMessage("끝")]))
    app = book_261618_graph(model)
    cfg = {"configurable": {"thread_id": "t2"}}
    try:
        app.invoke({"messages": [{"role": "user", "content": "검색하고 파일도 지워 줘"}]}, cfg)
        res = app.invoke(Command(resume="reject"), cfg)
        out("261618 mixed calls, sensitive rejected", tools_executed=list(RAN),
            tool_messages=[m.content for m in res["messages"] if isinstance(m, ToolMessage)])
    except Exception as e:  # noqa: BLE001
        out("261618 mixed calls, sensitive rejected", tools_executed=list(RAN),
            error=f"{type(e).__name__}: {str(e)[:300]}")


def run_rule1_try_except() -> None:
    """Page 261619 rule 1: interrupt() inside try/except Exception."""
    def bad_node(state):
        try:
            response = interrupt({"message": "승인?"})
            return {"messages": [AIMessage(content=str(response))]}
        except Exception as e:  # noqa: BLE001 - the book's 'wrong' example
            return {"messages": [AIMessage(content=f"에러 발생 ({type(e).__name__})")]}

    g = StateGraph(MessagesState)
    g.add_node("bad", bad_node)
    g.add_edge(START, "bad")
    g.add_edge("bad", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "r1"}}
    res = app.invoke({"messages": [{"role": "user", "content": "x"}]}, cfg)
    out("261619 rule 1: interrupt inside try/except Exception", paused="__interrupt__" in res,
        last=res["messages"][-1].content)


def run_rule4_rerun() -> None:
    """Page 261619 rule 4: code before interrupt() runs again on resume."""
    count = {"before": 0, "after": 0}

    def node(state):
        count["before"] += 1
        answer = interrupt("승인?")
        count["after"] += 1
        return {"messages": [AIMessage(content=str(answer))]}

    g = StateGraph(MessagesState)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "r4"}}
    app.invoke({"messages": [{"role": "user", "content": "x"}]}, cfg)
    app.invoke(Command(resume="approve"), cfg)
    out("261619 rule 4: code before interrupt on resume", before=count["before"], after=count["after"])


def run_no_checkpointer() -> None:
    def node(state):
        return {"messages": [AIMessage(content=str(interrupt("승인?")))]}

    g = StateGraph(MessagesState)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile()
    try:
        res = app.invoke({"messages": [{"role": "user", "content": "x"}]})
        out("interrupt without a checkpointer", ok=True, keys=sorted(res))
    except Exception as e:  # noqa: BLE001
        out("interrupt without a checkpointer", ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    for cfg in (None, {"configurable": {"thread_id": "nc"}}):
        try:
            res = app.invoke(Command(resume="approve"), cfg)
            out("resume without a checkpointer", config=cfg, ok=True, result=res)
        except Exception as e:  # noqa: BLE001
            out("resume without a checkpointer", config=cfg, ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    out("versions", langgraph=version("langgraph"), langchain_core=version("langchain-core"))
    run_261618_reject()
    run_261618_mixed_reject()
    run_rule1_try_except()
    run_rule4_rerun()
    run_no_checkpointer()
