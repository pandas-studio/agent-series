"""Why thinking-on runs of the interrupt question timed out: stream the first agent call and count what comes back.

Run from s1-langgraph/: uv run python ../runs/s1-e03/thinking_timeout_check.py  (output thinking-timeout-check.txt)
Stops reading after 240 s.
"""
import json
import os
import time
import urllib.request

from langchain_core.utils.function_calling import convert_to_openai_tool

from research_agent.agent import SYSTEM
from research_agent.config import load_settings
from research_agent.tools import search_docs

s = load_settings()
QUESTION = "interrupt로 멈춘 그래프를 Command(resume=...)로 다시 시작하는 방법은?"
body = {"model": s.model, "temperature": s.temperature, "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}],
        "tools": [convert_to_openai_tool(search_docs)],
        "chat_template_kwargs": {"enable_thinking": True}}
req = urllib.request.Request(f"{s.base_url}/chat/completions", json.dumps(body).encode(),
                             {"content-type": "application/json", "authorization": f"Bearer {s.api_key}"})
start = time.perf_counter()
reasoning, content, tool_args, finish, usage = [], [], [], None, None
with urllib.request.urlopen(req, timeout=60) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        usage = chunk.get("usage") or usage
        for c in chunk.get("choices") or []:
            d = c.get("delta") or {}
            reasoning.append(d.get("reasoning") or d.get("reasoning_content") or "")
            content.append(d.get("content") or "")
            for t in d.get("tool_calls") or []:
                tool_args.append((t.get("function") or {}).get("arguments") or "")
            finish = c.get("finish_reason") or finish
        if time.perf_counter() - start > 240:
            finish = finish or "stopped reading at 240 s"
            break
text = "".join(reasoning)
print(f"seconds {time.perf_counter() - start:.1f}  finish {finish}  usage {usage}")
print(f"reasoning chars {len(text)}  content chars {len(''.join(content))}  tool call args {''.join(tool_args)!r}")
print("reasoning start:", text[:300].replace("\n", " "))
print("reasoning end:", text[-600:].replace("\n", " "))
