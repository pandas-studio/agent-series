"""Route check: the classify step with each structured-output method, on labelled questions, run twice.

Run: uv run python -m research_agent.route_check > ../runs/s1-e02/route-check.jsonl.txt
Prints one JSON line per call and a summary line per method at the end.
"""

import json
import sys
import time

from .graph import CLASSIFY_PROMPT, Route
from .model import make_model

QUESTIONS = [
    ("LangGraph에서 checkpointer는 무엇을 저장하나요?", "docs"),
    ("interrupt로 멈춘 그래프를 Command(resume=...)로 다시 시작하는 방법은?", "docs"),
    ("StateGraph에서 조건부 엣지는 어떻게 추가하나요?", "docs"),
    ("LangChain create_agent에 도구를 넘기는 방법을 알려 주세요.", "docs"),
    ("파이썬에서 리스트를 거꾸로 뒤집는 방법은?", "direct"),
    ("HTTP 상태 코드 404는 무슨 뜻인가요?", "direct"),
    ("서울에서 부산까지 KTX로 얼마나 걸리나요?", "direct"),
    ("좋은 커밋 메시지를 쓰는 요령을 세 가지만 알려 주세요.", "direct"),
]
METHODS = ("json_schema", "function_calling", "json_mode")


def check(model, method: str, question: str, expected: str) -> dict:
    router = model.with_structured_output(Route, method=method, include_raw=True)
    start = time.perf_counter()
    try:
        out = router.invoke(CLASSIFY_PROMPT.format(question=question))
        parsed, error, raw = out["parsed"], out["parsing_error"], out["raw"]
    except Exception as e:  # a server-side refusal counts as a failed call
        parsed, error, raw = None, e, None
    seconds = round(time.perf_counter() - start, 2)
    got = parsed.route if parsed else None
    return {"method": method, "question": question, "expected": expected, "route": got, "ok": got == expected,
            "valid": parsed is not None, "seconds": seconds,
            "error": f"{type(error).__name__}: {str(error)[:160]}" if error else None,
            "tool_calls": len(getattr(raw, "tool_calls", None) or []) if raw is not None else None,
            "content": (getattr(raw, "content", "") or "")[:120] if raw is not None else None}


def main() -> int:
    model = make_model()
    rows = []
    for run in (1, 2):
        for method in METHODS:
            for question, expected in QUESTIONS:
                row = dict(check(model, method, question, expected), run=run)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    for method in METHODS:
        mine = [r for r in rows if r["method"] == method]
        print(json.dumps({"summary": method, "calls": len(mine), "valid": sum(r["valid"] for r in mine),
                          "correct": sum(r["ok"] for r in mine),
                          "mean_seconds": round(sum(r["seconds"] for r in mine) / len(mine), 2)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
