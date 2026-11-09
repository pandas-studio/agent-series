import os

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from research_agent.hello_graph import build


def test_one_node_graph_appends_model_reply():
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="안녕하세요")]))
    out = build(fake).invoke({"messages": [("user", "인사해 주세요")]})
    assert [m.type for m in out["messages"]] == ["human", "ai"]
    assert out["messages"][-1].content == "안녕하세요"


def test_graph_shape():
    fake = GenericFakeChatModel(messages=iter([]))
    g = build(fake).get_graph()
    assert set(g.nodes) == {"__start__", "answer", "__end__"}


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("AGENT_LIVE") != "1", reason="set AGENT_LIVE=1 to call the model server")
def test_live_graph_runs_on_local_model():
    out = build().invoke({"messages": [("user", "1 더하기 1은? 숫자만 답하세요.")]})
    assert "2" in out["messages"][-1].content
