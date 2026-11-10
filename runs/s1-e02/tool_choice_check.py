"""Why with_structured_output(method="function_calling") gets nothing back from vLLM + Qwen (s1-e02 notes).

Sends the Route tool straight to the server with each tool_choice, thinking off and on, and prints what comes back.
Run from s1-langgraph/: uv run python ../runs/s1-e02/tool_choice_check.py
"""
import json
from openai import OpenAI
from research_agent.graph import CLASSIFY_PROMPT, Route
c = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
tool = {"type": "function", "function": {"name": "Route", "description": Route.__doc__, "parameters": Route.model_json_schema()}}
msg = [{"role": "user", "content": CLASSIFY_PROMPT.format(question="StateGraph에서 조건부 엣지는 어떻게 추가하나요?")}]
for think in (False, True):
    for choice in ("auto", "required", {"type": "function", "function": {"name": "Route"}}):
        r = c.chat.completions.create(model="nvidia/Qwen3.6-35B-A3B-NVFP4", messages=msg, tools=[tool], tool_choice=choice,
                                      extra_body={"chat_template_kwargs": {"enable_thinking": think}}, max_tokens=600)
        m = r.choices[0].message
        calls = [(x.function.name, x.function.arguments[:70]) for x in (m.tool_calls or [])]
        reasoning = (m.model_dump().get("reasoning") or "")
        print("thinking", think, "| choice", "named" if isinstance(choice, dict) else choice, "| finish", r.choices[0].finish_reason,
              "| out_tokens", r.usage.completion_tokens, "| tool_calls", calls, "| content", repr((m.content or "")[:40]),
              "| reasoning chars", len(reasoning))
