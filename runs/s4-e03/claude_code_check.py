"""Claude Code as the MCP client of the docs server, headless (`claude -p`). Run from the repository root.

Each case runs `claude -p` once with --output-format stream-json and records what the session saw and did: whether
the docs server connected, which mcp__docs tools it offered, the tool calls, permission denials, the input tokens of
the first model request, turns, time and the start of the answer. Other MCP servers on this machine are not
recorded (--strict-mcp-config keeps them out, and the plain case only reports the docs server).

Every case runs with --permission-mode default, so the result does not depend on this machine's own default mode
(here "auto", whose classifier approved the tool calls in the first runs: claude-code-check-auto-mode.jsonl.txt).

Output: runs/s4-e03/claude-code-check.jsonl.txt   (python3 runs/s4-e03/claude_code_check.py [A B ...] for some cases)
"""
import json
import os
import subprocess
import sys
import time

MODEL = os.environ.get("CHECK_MODEL", "sonnet")
QUESTION = ("LangGraph에서 checkpointer는 무엇을 저장하나요? docs MCP 서버의 도구로 찾아서 3문장으로 답하고, "
            "사용한 페이지 경로를 적어 주세요.")
DOCS_TOOLS = "mcp__docs__search_docs mcp__docs__read_doc"
MCP = ["--mcp-config", ".mcp.json", "--strict-mcp-config"]

CASES = [
    ("A plain: .mcp.json pending approval", [], QUESTION, {}),
    ("B --mcp-config, no allowed tools", MCP, QUESTION, {}),
    ("C --mcp-config + --allowedTools", MCP + ["--allowedTools", DOCS_TOOLS], QUESTION, {}),
    ("D resource @-mention", MCP, "@docs:docs://page/langgraph/checkpointers.mdx 이 페이지를 3문장으로 요약해 주세요.", {}),
    ("E MCP prompt as a slash command", MCP + ["--allowedTools", DOCS_TOOLS],
     "/mcp__docs__ask_docs LangGraph에서 checkpointer는 무엇을 저장하나요?", {}),
    ("F no MCP server (baseline tokens)", ["--strict-mcp-config"], "안녕하세요 한 단어로만 답하세요.", {}),
    ("G docs server, tool search on (default)", MCP, "안녕하세요 한 단어로만 답하세요.", {}),
    ("H docs server, ENABLE_TOOL_SEARCH=false", MCP, "안녕하세요 한 단어로만 답하세요.", {"ENABLE_TOOL_SEARCH": "false"}),
    ("I no MCP server, ENABLE_TOOL_SEARCH=false", ["--strict-mcp-config"], "안녕하세요 한 단어로만 답하세요.",
     {"ENABLE_TOOL_SEARCH": "false"}),
]


def run(name: str, flags: list, prompt: str, env: dict) -> dict:
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--model", MODEL,
           "--max-turns", "10", "--permission-mode", os.environ.get("CHECK_PERMISSION_MODE", "default")] + flags
    start = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env={**os.environ, **env})
    seconds = round(time.perf_counter() - start, 1)
    init, calls, results, first_usage, final = None, [], [], None, {}
    for line in p.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            init = ev
        elif ev.get("type") == "assistant":
            msg = ev.get("message") or {}
            if first_usage is None and msg.get("usage"):
                first_usage = msg["usage"]
            calls += [{"name": b.get("name"), "input": b.get("input")} for b in msg.get("content") or []
                      if b.get("type") == "tool_use"]
        elif ev.get("type") == "user":
            for b in (ev.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    text = b.get("content")
                    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
                    results.append({"is_error": bool(b.get("is_error")), "start": text[:120]})
        elif ev.get("type") == "result":
            final = ev
    servers = [s for s in (init or {}).get("mcp_servers") or [] if s.get("name") == "docs"]
    tools = (init or {}).get("tools") or []
    u = first_usage or {}
    first_in = sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return {"case": name, "model": (init or {}).get("model"), "docs_server": servers,
            "docs_tools": [t for t in tools if t.startswith("mcp__docs__")],
            "slash_commands_docs": [c for c in (init or {}).get("slash_commands") or [] if "docs" in c],
            "tool_calls": calls, "tool_results": results,
            "permission_denials": [d.get("tool_name") for d in final.get("permission_denials") or []],
            "first_request_input_tokens": first_in, "num_turns": final.get("num_turns"),
            "subtype": final.get("subtype"), "seconds": seconds, "exit": p.returncode, "answer": (final.get("result") or "")[:300],
            "stderr": p.stderr.strip()[-200:]}


def main() -> int:
    only = sys.argv[1:]
    for name, flags, prompt, env in CASES:
        if only and name.split()[0] not in only:
            continue
        print(json.dumps(run(name, flags, prompt, env), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
