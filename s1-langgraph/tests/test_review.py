"""Episode 5: the review node pauses before every search; Command(resume=...) approves, edits or rejects it."""

import os

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from research_agent.agent import REJECTED, ReviewDecision, build_graph
from research_agent.chat import open_saver

live = pytest.mark.skipif(os.environ.get("AGENT_LIVE") != "1", reason="set AGENT_LIVE=1 to call the model server")


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class CountingModel(ScriptedModel):
    """Counts how often the graph calls the model."""
    calls: int = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        return super()._generate(*args, **kwargs)


def search_call(query: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "search_docs", "id": call_id, "args": {"query": query}}])


@pytest.fixture
def docs(tmp_path, monkeypatch):
    (tmp_path / "langgraph").mkdir()
    (tmp_path / "langgraph" / "persistence.mdx").write_text(
        "---\ntitle: Persistence\n---\nA checkpointer saves graph state at every step.\nUse a thread_id to resume.\n")
    (tmp_path / "langgraph" / "interrupts.mdx").write_text("# Interrupts\nPause with interrupt and resume with Command.\n")
    monkeypatch.setenv("DOCS_ROOT", str(tmp_path))
    return tmp_path


CFG = {"configurable": {"thread_id": "t1"}}
QUESTION = {"messages": [{"role": "user", "content": "checkpointer는 무엇을 저장하나요?"}]}


def paused_graph(model, saver=None):
    graph = build_graph(model, review=True, checkpointer=saver or InMemorySaver())
    out = graph.invoke(QUESTION, CFG)
    return graph, out


def test_pauses_before_the_search_with_the_pending_call(docs):
    model = CountingModel(messages=iter([search_call("checkpointer"), AIMessage(content="답")]))
    graph, out = paused_graph(model)
    [pending] = out["__interrupt__"]
    assert pending.value["tool_calls"] == [{"name": "search_docs", "args": {"query": "checkpointer"}, "id": "c1"}]
    assert pending.response_schema == ReviewDecision.model_json_schema()
    assert not any(isinstance(m, ToolMessage) for m in out["messages"])   # nothing searched yet
    assert graph.get_state(CFG).next == ("review",)
    assert model.calls == 1


def test_approve_runs_the_reviewed_call_and_the_model_once_more(docs):
    model = CountingModel(messages=iter([search_call("checkpointer"), AIMessage(content="체크포인터는 상태를 저장합니다.")]))
    graph, _ = paused_graph(model)
    out = graph.invoke(Command(resume={"action": "approve"}), CFG)
    [result] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert result.content.startswith("[langgraph/persistence.mdx]")
    assert out["messages"][-1].content == "체크포인터는 상태를 저장합니다."
    assert model.calls == 2   # the review node re-ran on resume, the agent node did not


def test_edit_replaces_the_query_in_the_same_message(docs):
    model = ScriptedModel(messages=iter([search_call("체크포인터 저장"), AIMessage(content="답")]))
    graph, before = paused_graph(model)
    ai_id = before["messages"][-1].id
    out = graph.invoke(Command(resume={"action": "edit", "query": "interrupt Command"}), CFG)
    ais = [m for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    assert len(ais) == 1 and ais[0].id == ai_id   # replaced, not appended
    assert ais[0].tool_calls[0]["args"] == {"query": "interrupt Command"}
    [result] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert result.content.startswith("[langgraph/interrupts.mdx]")


def test_reject_runs_no_search_and_tells_the_model(docs):
    model = ScriptedModel(messages=iter([search_call("checkpointer"), AIMessage(content="검색하지 않고 답합니다.")]))
    graph, _ = paused_graph(model)
    out = graph.invoke(Command(resume={"action": "reject", "reason": "no search needed"}), CFG)
    [refused] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert refused.status == "error"
    assert refused.content == REJECTED.format(reason="no search needed")
    assert out["messages"][-1].content == "검색하지 않고 답합니다."


def test_invalid_resume_value_is_refused_and_the_thread_stays_paused(docs):
    model = ScriptedModel(messages=iter([search_call("checkpointer"), AIMessage(content="답")]))
    graph, _ = paused_graph(model)
    with pytest.raises(ValidationError):
        graph.invoke(Command(resume={"action": "maybe"}), CFG)
    assert graph.get_state(CFG).next == ("review",)
    out = graph.invoke(Command(resume={"action": "approve"}), CFG)
    assert out["messages"][-1].content == "답"


def test_without_review_the_search_runs_without_a_pause(docs):
    model = ScriptedModel(messages=iter([search_call("checkpointer"), AIMessage(content="답")]))
    out = build_graph(model, checkpointer=InMemorySaver()).invoke(QUESTION, CFG)
    assert "__interrupt__" not in out
    assert any(isinstance(m, ToolMessage) for m in out["messages"])


def test_a_paused_thread_resumes_from_another_saver_on_the_same_file(docs, tmp_path):
    db = tmp_path / "chat.sqlite"
    with open_saver(db) as saver:
        _, out = paused_graph(ScriptedModel(messages=iter([search_call("checkpointer")])), saver)
        assert "__interrupt__" in out
    with open_saver(db) as saver:   # a new process would open the file the same way
        graph = build_graph(ScriptedModel(messages=iter([AIMessage(content="재개한 답")])), review=True,
                            checkpointer=saver)
        assert graph.get_state(CFG).next == ("review",)
        out = graph.invoke(Command(resume={"action": "approve"}), CFG)
    assert out["messages"][-1].content == "재개한 답"


@pytest.mark.live
@live
def test_live_review_pauses_then_approve_answers():
    graph = build_graph(guard=True, review=True, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "live-review"}}
    out = graph.invoke({"messages": [{"role": "user", "content": "LangGraph에서 checkpointer는 무엇을 저장하나요?"}]}, cfg)
    assert out["__interrupt__"][0].value["tool_calls"][0]["name"] == "search_docs"
    out = graph.invoke(Command(resume={"action": "approve"}), cfg)
    assert any(isinstance(m, ToolMessage) for m in out["messages"])
    while "__interrupt__" in out:   # a second search pauses again
        out = graph.invoke(Command(resume={"action": "approve"}), cfg)
    assert out["messages"][-1].content
