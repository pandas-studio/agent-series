"""Thinking on: how long does one agent step generate? The same input sent 6 times (run from s1-langgraph/).

Builds the agent's input right after its last search_docs result for the interrupt question (thinking on; before the
MDX import fix that was the second of two searches, as in the traced run), then sends that exact request 6 times, streaming, without retries, and counts the generated tokens and time.
Stops reading a response after 180 s. Output: ../runs/s1-e03/thinking-runaway-check.txt
"""
import json
import time
import urllib.request

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models.base import _convert_message_to_dict

from research_agent.agent import SYSTEM, build_graph
from research_agent.config import load_settings
from research_agent.model import make_model
from research_agent.tools import search_docs

QUESTION = "interrupt로 멈춘 그래프를 Command(resume=...)로 다시 시작하는 방법은?"
s = load_settings({"AGENT_THINKING": "1", "AGENT_TIMEOUT": "300"})

messages = None
for state in build_graph(make_model(s)).stream({"messages": [{"role": "user", "content": QUESTION}]},
                                               stream_mode="values"):
    if isinstance(state["messages"][-1], ToolMessage):
        messages = state["messages"]   # keep the latest input that ends with a search result
if messages is None:
    raise SystemExit("the agent answered without searching; rerun")
print("input after", [type(m).__name__ for m in messages],
      "queries", [c["args"]["query"] for m in messages for c in getattr(m, "tool_calls", []) or []])

body = {"model": s.model, "temperature": s.temperature, "stream": True, "stream_options": {"include_usage": True},
        "messages": [_convert_message_to_dict(m) for m in [SystemMessage(SYSTEM)] + messages],
        "tools": [convert_to_openai_tool(search_docs)], "chat_template_kwargs": {"enable_thinking": True}}
for attempt in range(1, 7):
    req = urllib.request.Request(f"{s.base_url}/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json", "authorization": f"Bearer {s.api_key}"})
    start, chunks, finish, usage, tool, text = time.perf_counter(), 0, None, None, "", []
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            usage = c.get("usage") or usage
            for ch in c.get("choices") or []:
                chunks += 1
                finish = ch.get("finish_reason") or finish
                delta = ch.get("delta") or {}
                text.append(delta.get("reasoning") or delta.get("reasoning_content") or delta.get("content") or "")
                for t in delta.get("tool_calls") or []:
                    tool += (t.get("function") or {}).get("arguments") or ""
            if time.perf_counter() - start > 180:
                finish = finish or "still generating at 180 s"
                break
    took = time.perf_counter() - start
    out = (usage or {}).get("completion_tokens") or chunks
    print(f"attempt {attempt}: {took:6.1f} s, ~{out} tokens ({out / took:.0f} tok/s), finish {finish}, "
          f"tool args {tool[:60]!r}", flush=True)
    if finish != "tool_calls" and finish != "stop":
        tail = "".join(text)[-700:].replace("\n", " ")
        print(f"  last 700 characters: {tail}", flush=True)
