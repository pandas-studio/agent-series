"""Does the local Qwen use the docs server's MCP tools in create_agent? One pass = 2 prompts × 4 questions × 3.

- prompt "s4": docs_mcp.agent.SYSTEM (docs-only assistant: search, read, answer with the path);
- prompt "s1": episode s1-e03's SYSTEM, copied verbatim (search first for LangGraph/LangChain questions, answer
  other questions directly, 3-5 sentences). In s1-e03 create_agent with that prompt and a local @tool searched
  2 of 24 times.

The tools are discovered once per pass through MCPAdapter (stdio, counting_server.py) and the agent runs after the
adapter's `async with` has exited, as in docs_mcp.agent.ask. Records tool calls, model calls, error tool messages,
seconds, whether the expected page path is in the answer, and the server starts of the pass.

Run from s4-mcp: AGENT_LIVE=1 uv run python ../runs/s4-e04/agent_check.py > ../runs/s4-e04/agent-check-r1.jsonl.txt
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from docs_mcp.agent import SYSTEM as S4_SYSTEM
from docs_mcp.agent import MCPAdapter, make_model
from langchain.agents import create_agent

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "counting_server.py"
STARTS = SCRIPT.with_suffix(".starts")

S1_SYSTEM = """You answer questions for a developer.
If the question is about LangGraph or LangChain, call search_docs first and answer only from the pages it returns,
naming the page paths you used. For any other question, answer directly without tools.
Answer in Korean, in 3-5 sentences."""

QUESTIONS = {   # s1-e03 tool_check questions and their expected pages
    "LangGraph에서 checkpointer는 무엇을 저장하나요?": "langgraph/checkpointers.mdx",
    "interrupt로 멈춘 그래프를 Command(resume=...)로 다시 시작하는 방법은?": "langgraph/interrupts.mdx",
    "StateGraph에서 조건부 엣지는 어떻게 추가하나요?": "langgraph/graph-api-control-flow.mdx",
    "LangChain create_agent에 도구를 넘기는 방법을 알려 주세요.": "langchain/agents.mdx",
}
REPEATS = 3


async def one(agent, prompt: str, question: str, expected: str) -> dict:
    t0 = time.perf_counter()
    try:
        out = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    except Exception as e:  # noqa: BLE001 - a failed run is a measurement too
        return {"prompt": prompt, "question": question, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "seconds": round(time.perf_counter() - t0, 2)}
    msgs = out["messages"]
    calls = [c for m in msgs if isinstance(m, AIMessage) for c in m.tool_calls]
    answer = msgs[-1].content if isinstance(msgs[-1].content, str) else str(msgs[-1].content)
    return {"prompt": prompt, "question": question,
            "searched": any(c["name"] == "search_docs" for c in calls),
            "tool_calls": [c["name"] for c in calls],
            "queries": [c["args"].get("query") for c in calls if c["name"] == "search_docs"],
            "read_paths": [c["args"].get("path") for c in calls if c["name"] == "read_doc"],
            "tool_errors": [m.text[:120] for m in msgs if isinstance(m, ToolMessage) and m.status == "error"],
            "model_calls": sum(isinstance(m, AIMessage) for m in msgs),
            "expected_in_answer": expected in answer, "answer_chars": len(answer),
            "seconds": round(time.perf_counter() - t0, 2), "answer": answer[:160]}


async def main() -> int:
    STARTS.unlink(missing_ok=True)
    config = {"mcpServers": {"docs": {"command": sys.executable, "args": [str(SCRIPT)]}}}
    async with MCPAdapter(config) as adapter:
        tools = await adapter.list_tools()
    model = make_model()
    rows = []
    for prompt, system in (("s4", S4_SYSTEM), ("s1", S1_SYSTEM)):
        agent = create_agent(model, tools, system_prompt=system)
        for _ in range(REPEATS):
            for question, expected in QUESTIONS.items():
                row = await one(agent, prompt, question, expected)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    starts = len(STARTS.read_text().splitlines()) if STARTS.exists() else 0
    for prompt in ("s4", "s1"):
        mine = [r for r in rows if r["prompt"] == prompt and "error" not in r]
        print(json.dumps({"summary": prompt, "runs": len(mine), "searched": sum(r["searched"] for r in mine),
                          "read_doc_runs": sum("read_doc" in r["tool_calls"] for r in mine),
                          "expected_in_answer": sum(r["expected_in_answer"] for r in mine),
                          "mean_model_calls": round(sum(r["model_calls"] for r in mine) / max(1, len(mine)), 2),
                          "mean_seconds": round(sum(r["seconds"] for r in mine) / max(1, len(mine)), 2),
                          "errors": sum(1 for r in rows if r["prompt"] == prompt and "error" in r)},
                         ensure_ascii=False), flush=True)
    print(json.dumps({"summary": "pass", "server_starts": starts}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
