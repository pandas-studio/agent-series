import os

import pytest

from research_agent import Settings, load_settings, make_model
from research_agent import check

live = pytest.mark.skipif(os.environ.get("AGENT_LIVE") != "1", reason="set AGENT_LIVE=1 to call the model server")


def test_defaults_point_at_local_server():
    s = load_settings({})
    assert s.base_url == "http://localhost:8000/v1"
    assert s.thinking is False


def test_env_overrides():
    s = load_settings({"AGENT_BASE_URL": "http://gpu:9000/v1", "AGENT_MODEL": "m", "AGENT_THINKING": "yes",
                       "AGENT_TEMPERATURE": "0"})
    assert (s.base_url, s.model, s.thinking, s.temperature) == ("http://gpu:9000/v1", "m", True, 0.0)


def test_model_sends_thinking_switch():
    m = make_model(Settings(thinking=False))
    assert m.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert m.openai_api_base == "http://localhost:8000/v1"


@pytest.mark.live
@live
def test_live_check_answers_in_korean():
    r = check.run()
    assert r["output_tokens"] > 0
    assert any("가" <= ch <= "힣" for ch in r["answer"])  # Hangul in the answer


@pytest.mark.live
@live
def test_live_thinking_off_is_short():
    # With thinking off, a one-sentence answer should not spend hundreds of tokens on reasoning.
    r = check.run()
    assert r["output_tokens"] < 200
