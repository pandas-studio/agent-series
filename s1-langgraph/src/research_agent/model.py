"""Chat model factory: one place that knows how to reach the local model."""

from langchain_openai import ChatOpenAI

from .config import Settings, load_settings


def make_model(settings: Settings | None = None) -> ChatOpenAI:
    s = settings or load_settings()
    return ChatOpenAI(
        model=s.model,
        base_url=s.base_url,
        api_key=s.api_key,
        temperature=s.temperature,
        timeout=s.timeout,
        # vLLM passes chat_template_kwargs to the Qwen chat template.
        extra_body={"chat_template_kwargs": {"enable_thinking": s.thinking}},
    )
