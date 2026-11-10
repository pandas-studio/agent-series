import os

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from research_agent.docs import search
from research_agent.graph import Route, State, build, make_router

live = pytest.mark.skipif(os.environ.get("AGENT_LIVE") != "1", reason="set AGENT_LIVE=1 to call the model server")


@pytest.fixture
def docs(tmp_path):
    (tmp_path / "langgraph").mkdir()
    (tmp_path / "langgraph" / "persistence.mdx").write_text(
        "---\ntitle: Persistence\n---\nA checkpointer saves graph state at every step.\nUse a thread_id to resume.\n")
    (tmp_path / "langgraph" / "streaming.mdx").write_text("# Streaming\nStream graph state as it runs.\n")
    return tmp_path


def router_for(route: str):
    """A stand-in for the structured-output router: always returns this route."""
    return RunnableLambda(lambda _prompt: Route(route=route, reason="test"))


def model_saying(*replies: str):
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def test_docs_question_goes_through_retrieve(docs):
    out = build(model_saying("체크포인터는 단계마다 상태를 저장합니다."), router_for("docs"), docs).invoke(
        {"question": "checkpointer는 무엇을 저장하나요?", "steps": []})
    assert out["steps"] == ["classify", "retrieve", "answer"]
    assert [h["path"] for h in out["hits"]] == ["langgraph/persistence.mdx"]
    assert out["answer"].startswith("체크포인터는")


def test_direct_question_skips_retrieve(docs):
    out = build(model_saying("404는 찾을 수 없다는 뜻입니다."), router_for("direct"), docs).invoke(
        {"question": "HTTP 404는 무슨 뜻인가요?", "steps": []})
    assert out["steps"] == ["classify", "answer"] and "hits" not in out


def test_steps_reducer_appends_instead_of_replacing(docs):
    # each node returns only ["<its name>"]; the operator.add reducer on State.steps joins them
    assert State.__annotations__["steps"].__metadata__[0].__name__ == "add"
    out = build(model_saying("답"), router_for("docs"), docs).invoke({"question": "checkpointer", "steps": ["start"]})
    assert out["steps"] == ["start", "classify", "retrieve", "answer"]


def test_graph_shape_has_a_conditional_edge(docs):
    g = build(model_saying(), router_for("docs"), docs).get_graph()
    assert set(g.nodes) == {"__start__", "classify", "retrieve", "answer", "__end__"}
    conditional = {(e.source, e.target) for e in g.edges if e.conditional}
    assert conditional == {("classify", "retrieve"), ("classify", "answer")}


def test_search_uses_english_words_only(docs):
    assert [h["path"] for h in search("thread_id로 다시 시작", root=docs)] == ["langgraph/persistence.mdx"]
    assert search("한국어만 있는 질문", root=docs) == []


def test_search_ignores_words_on_every_page(docs):
    # "graph" is on both pages, so only "checkpointer" decides; plain counts would favour the page that repeats "graph"
    (docs / "langgraph" / "streaming.mdx").write_text("# Streaming\n" + "graph " * 50)
    assert [h["path"] for h in search("graph checkpointer", root=docs)] == ["langgraph/persistence.mdx"]
    assert search("graph", root=docs) == []   # a word on every page finds nothing


@pytest.mark.live
@live
@pytest.mark.parametrize("question, route", [("LangGraph에서 checkpointer는 무엇을 저장하나요?", "docs"),
                                             ("HTTP 상태 코드 404는 무슨 뜻인가요?", "direct")])
def test_live_router_with_json_schema(question, route):
    from research_agent.graph import CLASSIFY_PROMPT
    from research_agent.model import make_model
    assert make_router(make_model()).invoke(CLASSIFY_PROMPT.format(question=question)).route == route


@pytest.mark.live
@live
def test_live_docs_question_cites_a_page():
    out = build().invoke({"question": "LangGraph에서 checkpointer는 무엇을 저장하나요?", "steps": []})
    assert out["steps"] == ["classify", "retrieve", "answer"]
    # the first run ranked graph-api-control-flow.mdx first ("langgraph" 229 times) and the answer said the excerpts
    # did not cover it, while a check for "some page path in the answer" still passed
    assert out["hits"][0]["path"] == "langgraph/checkpointers.mdx"
    assert "langgraph/checkpointers.mdx" in out["answer"]
