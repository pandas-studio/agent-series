# s1-e01 — project start, local model connection (2026-11-09)

Facts recorded while building this episode. This is source material for the blog pipeline, not post text.

## Setup
- Machine: NVIDIA DGX Spark (GB10, 128 GB unified memory), Ubuntu 24.04.
- Model server: vLLM 0.21.0 serving `nvidia/Qwen3.6-35B-A3B-NVFP4` on http://localhost:8000/v1 (OpenAI-compatible,
  no API key, max_model_len 262144).
- Packages (uv, Python 3.12): langgraph 1.2.20, langchain-core 1.7.6, langchain-openai 1.7.4, openai 3.46.0
  (`versions.txt`).

## How Claude Code was used
- The series plan (4 series, 30 episodes) was made in plan mode first; this episode's code was written after that
  plan was approved, in auto mode.
- Before writing code, Claude Code sent one raw HTTP chat request with a `tools` list to the server, to confirm that
  vLLM returns `tool_calls` for Qwen (it did: `get_weather({"city": "Seoul"})`, 27 output tokens). Later episodes
  depend on tool calling, so this was checked first.
- `uv init --lib` filled `authors` with the git user's e-mail. Claude Code removed it before the first commit,
  because the repo may become public.
- `uv init` also wrote `readme = "README.md"`; after README.md was deleted the build failed with
  "failed to open file .../README.md". Fix: remove the `readme` line.
- CLAUDE.md for the project lists the environment, the test commands and the rules (model passed as a parameter,
  live output saved under runs/, APIs checked against official docs).

## Design decisions
- All server settings come from `AGENT_*` environment variables (`config.py`); one factory (`model.py`) builds the
  `ChatOpenAI` client with `base_url` pointed at the local server. Switching to another OpenAI-compatible server
  (llama.cpp, Ollama, a cloud API) changes only environment variables.
- Thinking off by default: `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`. vLLM passes it to the
  Qwen chat template.
- Two test layers: unit tests use `GenericFakeChatModel` (offline, 0.5 s); `live` tests run only with
  `AGENT_LIVE=1` and call the real server.
- `hello_graph.py` is a one-node graph (START → answer → END) on `MessagesState`, to show that the same model
  object plugs into LangGraph unchanged. Episode 2 replaces it.

## Measurements (`check-thinking.jsonl.txt`; same one-sentence Korean question, 2 runs each, 11-09)
| thinking | seconds | output tokens | tokens/s |
| --- | --- | --- | --- |
| off | 0.65, 0.90 | 43, 62 | 66.2, 69.0 |
| on | 7.91, 8.81 | 580, 645 | 73.3, 73.2 |

An earlier pair of runs the same evening gave off 0.56–0.92 s / 36–63 tokens and on 5.39–8.65 s / 394–634 tokens.
With thinking on, the answer is the same length, but the model spends roughly 10× the tokens and time before it.
Generation speed itself barely changes (66–73 tok/s).

## Difference from the author's book
- LangGraph book 1-2-2 (page 261588) sets up an OpenAI GPT model with an API key. Here the same `ChatOpenAI` class
  talks to a local server through `base_url`; no key is needed.
- The Qwen thinking switch (`chat_template_kwargs`) is server-specific and not in the book.

## Tests
- Unit: 5 passed, 3 skipped (`pytest-unit.txt`). Live: 3 passed (`pytest-live.txt`).
