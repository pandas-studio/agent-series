"""Connection check: send one question to the model and report time and tokens.

Run: uv run python -m research_agent.check
"""

import json
import sys
import time

from .config import load_settings
from .model import make_model

QUESTION = "LangGraph의 StateGraph가 무엇인지 한 문장으로 설명해 주세요."


def run(question: str = QUESTION) -> dict:
    settings = load_settings()
    model = make_model(settings)
    start = time.perf_counter()
    reply = model.invoke(question)
    elapsed = time.perf_counter() - start
    usage = reply.usage_metadata or {}
    out_tokens = usage.get("output_tokens", 0)
    return {
        "base_url": settings.base_url,
        "model": settings.model,
        "thinking": settings.thinking,
        "seconds": round(elapsed, 2),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": out_tokens,
        "tokens_per_second": round(out_tokens / elapsed, 1) if elapsed else None,
        "answer": reply.content,
    }


def main() -> int:
    try:
        result = run(" ".join(sys.argv[1:]) or QUESTION)
    except Exception as e:  # report any connection failure in one line
        print(f"connection failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
