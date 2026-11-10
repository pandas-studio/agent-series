"""Can the first model call be made to search? tool_choice auto / required / named, thinking off (run from s1-langgraph/).

The 4 docs questions, once each per tool_choice; first model call only. Output: ../runs/s1-e03/force-tool-check.txt
"""
import time

from langchain_core.messages import SystemMessage

from research_agent.agent import SYSTEM
from research_agent.model import make_model
from research_agent.tool_check import EXPECTED
from research_agent.tools import search_docs

model = make_model()
for choice in ("auto", "required", "search_docs"):
    llm = model.bind_tools([search_docs], tool_choice=choice)
    for q in EXPECTED:
        start = time.perf_counter()
        try:
            r = llm.invoke([SystemMessage(SYSTEM), ("user", q)])
            calls = [c["args"] for c in r.tool_calls]
            u = r.usage_metadata or {}
            print(f"{choice:12} {time.perf_counter() - start:5.2f} s  out {u.get('output_tokens')}  "
                  f"finish {r.response_metadata.get('finish_reason')}  tool_calls {calls}  content {r.content[:50]!r}  | {q[:30]}",
                  flush=True)
        except Exception as e:
            print(f"{choice:12} ERROR {type(e).__name__}: {str(e)[:120]} | {q[:30]}", flush=True)
