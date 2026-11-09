# S1 research agent — rules for Claude Code

## What this is
A LangGraph research agent that answers from the local doc pages in `../corpus/`.
Each blog episode adds one capability. The episode's state is tagged `s1-eNN`.

## Environment
- Model: OpenAI-compatible server, default vLLM at http://localhost:8000/v1 (Qwen3.6). Settings come only from
  `AGENT_*` environment variables (`src/research_agent/config.py`). Do not hard-code another URL or key.
- Thinking is off by default (`AGENT_THINKING=0`). Turn it on only for a step that needs it, and measure the cost.
- Python 3.12 with uv. Run `uv run pytest -q` for unit tests; `AGENT_LIVE=1 uv run pytest -q -m live` for tests
  that call the model.

## Rules
- Every model-facing function takes the model as a parameter, so unit tests can pass a fake
  (`GenericFakeChatModel`) and run offline.
- A change is done only when both the unit tests and the live tests pass. Save the live output under
  `../runs/s1-eNN/`.
- Check APIs against the current official docs (docs.langchain.com), not memory. When the API differs from the
  author's book, note the difference in `../runs/s1-eNN/notes.md`.
- Code and comments in English. Model prompts in English; user-facing sample questions may be Korean.
- Never commit secrets or the downloaded corpus pages.

## Episode notes (`../runs/s1-eNN/notes.md`)
- Start with `## Key finding`: 2-3 sentences on the one result the post opens with, including its number.
- List the packages added directly (`uv add …`) apart from the installed dependency versions.
- Then: Commands run, How Claude Code was used, Design decisions, Measurements, Difference from the author's book, Tests.
