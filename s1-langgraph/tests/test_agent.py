import os

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.checkpoint.memory import InMemorySaver

from research_agent.agent import MAX_MODEL_CALLS, REMINDER, build_graph, build_prebuilt, summary, this_turn
from research_agent.graph import Route
from research_agent.tools import VAGUE_DOC, search_docs, with_description

live = pytest.mark.skipif(os.environ.get("AGENT_LIVE") != "1", reason="set AGENT_LIVE=1 to call the model server")


class ScriptedModel(GenericFakeChatModel):
    """Replies from a script; bind_tools returns itself (the base class raises NotImplementedError)."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*replies):
    return ScriptedModel(messages=iter(replies))


def search_call(query: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "search_docs", "id": call_id, "args": {"query": query}}])


@pytest.fixture
def docs(tmp_path, monkeypatch):
    (tmp_path / "langgraph").mkdir()
    (tmp_path / "langgraph" / "persistence.mdx").write_text(
        "---\ntitle: Persistence\n---\nA checkpointer saves graph state at every step.\nUse a thread_id to resume.\n")
    (tmp_path / "langgraph" / "control.mdx").write_text("# Control flow\nAdd conditional edges with add_conditional_edges.\n")
    monkeypatch.setenv("DOCS_ROOT", str(tmp_path))
    return tmp_path


def test_tool_schema_and_description():
    assert search_docs.name == "search_docs"
    assert list(search_docs.args) == ["query"]
    assert "English keywords" in search_docs.description
    vague = with_description(VAGUE_DOC)
    assert vague.description == "Search the docs." and vague.args == search_docs.args
    assert "English keywords" in search_docs.description   # the original tool is unchanged


def test_tool_returns_paths_and_excerpts(docs):
    out = search_docs.invoke({"query": "conditional edges"})
    assert out.startswith("[langgraph/control.mdx] Control flow")
    assert search_docs.invoke({"query": "조건부 엣지"}) == "No page matched '조건부 엣지'."   # Korean words match nothing


@pytest.mark.parametrize("build", [build_graph, build_prebuilt])
def test_tool_call_loop(docs, build):
    model = scripted(search_call("checkpointer thread_id"), AIMessage(content="체크포인터는 상태를 저장합니다."))
    out = build(model).invoke({"messages": [{"role": "user", "content": "checkpointer는 무엇을 저장하나요?"}]})
    kinds = [type(m).__name__ for m in out["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    s = summary(out["messages"])
    assert s["queries"] == ["checkpointer thread_id"] and s["pages"] == [["langgraph/persistence.mdx"]]
    assert s["model_calls"] == 2 and s["answer"].startswith("체크포인터는")


@pytest.mark.parametrize("build", [build_graph, build_prebuilt])
def test_no_tool_call_ends_after_one_model_call(docs, build):
    out = build(scripted(AIMessage(content="404는 찾을 수 없다는 뜻입니다."))).invoke(
        {"messages": [{"role": "user", "content": "HTTP 404는?"}]})
    assert [type(m).__name__ for m in out["messages"]] == ["HumanMessage", "AIMessage"]
    assert summary(out["messages"])["queries"] == []


def test_graph_answers_without_tools_after_max_model_calls(docs):
    # before the final node, a run that hit the cap ended on a tool call: the answer was ""
    model = scripted(*[search_call("checkpointer", f"c{i}") for i in range(MAX_MODEL_CALLS)],
                     AIMessage(content="찾은 페이지로 답합니다."))
    out = build_graph(model).invoke({"messages": [{"role": "user", "content": "checkpointer"}]})
    assert summary(out["messages"])["model_calls"] == MAX_MODEL_CALLS + 1
    assert sum(isinstance(m, ToolMessage) for m in out["messages"]) == MAX_MODEL_CALLS
    s = summary(out["messages"])
    assert s["capped"] and s["answer"] == "찾은 페이지로 답합니다."


def test_graph_shape(docs):
    g = build_graph(scripted()).get_graph()
    assert set(g.nodes) == {"__start__", "agent", "tools", "final", "__end__"}
    assert {(e.source, e.target) for e in g.edges if e.conditional} == {
        ("agent", "tools"), ("agent", "__end__"), ("tools", "agent"), ("tools", "final")}


def router_for(route: str):
    return RunnableLambda(lambda _prompt: Route(route=route, reason="test"))


def test_guard_reminds_once_when_a_docs_question_is_answered_without_search(docs):
    model = scripted(AIMessage(content="기억으로 답합니다."), search_call("checkpointer"), AIMessage(content="문서로 답합니다."))
    out = build_graph(model, guard=True, router=router_for("docs")).invoke(
        {"messages": [{"role": "user", "content": "checkpointer는 무엇을 저장하나요?"}]})
    kinds = [type(m).__name__ for m in out["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert out["messages"][2].content == REMINDER and set(out) == {"messages", "route"}
    s = summary(out["messages"])
    assert s["reminded"] and s["queries"] == ["checkpointer"] and s["answer"] == "문서로 답합니다."


def test_guard_reminds_only_once(docs):
    model = scripted(AIMessage(content="기억으로 답합니다."), AIMessage(content="또 기억으로 답합니다."))
    out = build_graph(model, guard=True, router=router_for("docs")).invoke(
        {"messages": [{"role": "user", "content": "checkpointer"}]})
    assert sum(isinstance(m, HumanMessage) and m.content == REMINDER for m in out["messages"]) == 1
    assert out["messages"][-1].content == "또 기억으로 답합니다."


def test_guard_leaves_direct_questions_alone(docs):
    out = build_graph(scripted(AIMessage(content="404는 찾을 수 없다는 뜻입니다.")), guard=True,
                      router=router_for("direct")).invoke({"messages": [{"role": "user", "content": "HTTP 404는?"}]})
    assert [type(m).__name__ for m in out["messages"]] == ["HumanMessage", "AIMessage"] and out["route"] == "direct"


def test_guard_graph_shape(docs):
    g = build_graph(scripted(), guard=True, router=router_for("docs")).get_graph()
    assert set(g.nodes) == {"__start__", "classify", "agent", "tools", "remind", "final", "__end__"}
    assert {(e.source, e.target) for e in g.edges if e.conditional} == {
        ("agent", "tools"), ("agent", "remind"), ("agent", "__end__"), ("tools", "agent"), ("tools", "final")}


def test_this_turn_starts_at_the_last_user_question():
    msgs = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2"), AIMessage("no search"), HumanMessage(REMINDER)]
    question, turn = this_turn(msgs)
    assert question == "q2" and [m.content for m in turn] == ["q2", "no search", REMINDER]


def test_second_turn_on_a_thread_gets_its_own_reminder_cap_and_route(docs):
    """Episode 4: with a checkpointer, turn 2 used to inherit model_calls and reminded from turn 1 and was routed by
    the first question of the thread."""
    seen = []

    def route_last(prompt):
        seen.append(prompt)
        return Route(route="docs", reason="test")

    model = scripted(
        AIMessage(content="기억 1"), search_call("checkpointer", "c1"), AIMessage(content="문서 1"),   # turn 1
        AIMessage(content="기억 2"), search_call("thread_id", "c2"), AIMessage(content="문서 2"))      # turn 2
    g = build_graph(model, guard=True, router=RunnableLambda(route_last), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    g.invoke({"messages": [{"role": "user", "content": "첫 질문 checkpointer"}]}, cfg)
    out = g.invoke({"messages": [{"role": "user", "content": "둘째 질문 thread_id"}]}, cfg)
    q, turn = this_turn(out["messages"])
    assert q == "둘째 질문 thread_id"
    assert [type(m).__name__ for m in turn] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage",
                                                 "ToolMessage", "AIMessage"]   # reminded again, searched, answered
    assert turn[2].content == REMINDER and turn[-1].content == "문서 2"
    assert "둘째 질문 thread_id" in seen[1] and "첫 질문" not in seen[1]   # routed by this turn's question


def test_threads_do_not_share_messages(docs):
    g = build_graph(scripted(AIMessage(content="a"), AIMessage(content="b")), checkpointer=InMemorySaver())
    g.invoke({"messages": [{"role": "user", "content": "one"}]}, {"configurable": {"thread_id": "x"}})
    out = g.invoke({"messages": [{"role": "user", "content": "two"}]}, {"configurable": {"thread_id": "y"}})
    assert [m.content for m in out["messages"]] == ["two", "b"]


def test_sqlite_connect_without_check_same_thread_fails_on_invoke(docs, tmp_path):
    # LangGraph writes checkpoints from a worker thread; the docs' sqlite3.connect(path) form breaks on the first invoke
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    g = build_graph(scripted(AIMessage(content="a")), checkpointer=SqliteSaver(sqlite3.connect(tmp_path / "x.db")))
    with pytest.raises(sqlite3.ProgrammingError, match="same thread"):
        g.invoke({"messages": [{"role": "user", "content": "q"}]}, {"configurable": {"thread_id": "t"}})


def test_chat_thread_survives_a_new_saver(docs, tmp_path):
    from research_agent.chat import open_saver
    db = tmp_path / "chat.sqlite"
    cfg = {"configurable": {"thread_id": "demo"}}
    with open_saver(db) as saver:
        build_graph(scripted(AIMessage(content="첫 답")), checkpointer=saver).invoke(
            {"messages": [{"role": "user", "content": "첫 질문"}]}, cfg)
    with open_saver(db) as saver:   # a new connection, as in a new process
        out = build_graph(scripted(AIMessage(content="둘째 답")), checkpointer=saver).invoke(
            {"messages": [{"role": "user", "content": "둘째 질문"}]}, cfg)
    assert [m.content for m in out["messages"]] == ["첫 질문", "첫 답", "둘째 질문", "둘째 답"]


@pytest.mark.live
@live
def test_live_guard_makes_the_docs_question_search():
    out = build_graph(guard=True).invoke({"messages": [{"role": "user", "content": "StateGraph에서 조건부 엣지는 어떻게 추가하나요?"}]})
    s = summary(out["messages"])
    assert out["route"] == "docs" and s["queries"], "no search even after the reminder"
