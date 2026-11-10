import os

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from research_agent.agent import MAX_MODEL_CALLS, REMINDER, build_graph, build_prebuilt, summary
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
    assert out["model_calls"] == MAX_MODEL_CALLS + 1
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
    assert out["messages"][2].content == REMINDER and out["reminded"] is True
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


@pytest.mark.live
@live
def test_live_guard_makes_the_docs_question_search():
    out = build_graph(guard=True).invoke({"messages": [{"role": "user", "content": "StateGraph에서 조건부 엣지는 어떻게 추가하나요?"}]})
    s = summary(out["messages"])
    assert out["route"] == "docs" and s["queries"], "no search even after the reminder"
