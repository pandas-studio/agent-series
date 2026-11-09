"""Settings for the local model server, read from the environment.

Defaults point at a vLLM server on this machine. Override them to use any
OpenAI-compatible server (llama.cpp, Ollama, a cloud endpoint).
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://localhost:8000/v1"
    model: str = "nvidia/Qwen3.6-35B-A3B-NVFP4"
    # vLLM accepts any key unless it was started with --api-key.
    api_key: str = "local"
    temperature: float = 0.6
    # Qwen3 models think before answering by default. Agent steps are short,
    # so thinking is off unless asked for.
    thinking: bool = False
    timeout: float = 120.0


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(env: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    base = Settings()
    return Settings(
        base_url=env.get("AGENT_BASE_URL", base.base_url),
        model=env.get("AGENT_MODEL", base.model),
        api_key=env.get("AGENT_API_KEY", base.api_key),
        temperature=float(env.get("AGENT_TEMPERATURE", base.temperature)),
        thinking=_bool(env.get("AGENT_THINKING", str(base.thinking))),
        timeout=float(env.get("AGENT_TIMEOUT", base.timeout)),
    )
