# agent-series

블로그 연재의 예제 코드입니다. 모든 예제는 Claude Code로 만들었고, DGX Spark의 로컬 모델(vLLM, Qwen3.6)로 실행합니다.

| 폴더 | 시리즈 |
| --- | --- |
| `s1-langgraph/` | S1. Claude Code로 만드는 LangGraph 에이전트 |
| `s2-harness/` | S2. 하네스 해부: Claude Code와 Deep Agents |
| `s3-local-llm/` | S3. 로컬 LLM 에이전트 실전기 |
| `s4-mcp/` | S4. MCP로 에이전트 확장하기 |

각 회차가 끝난 상태에는 `s1-e01`과 같은 태그가 붙어 있습니다. 실행 기록은 `runs/<회차>/`에 있습니다.

## 준비

```sh
corpus/fetch.sh            # 예제가 읽는 공식 문서 페이지(langchain-ai/docs, MIT)
cd s1-langgraph && uv run pytest -q
AGENT_LIVE=1 uv run pytest -q -m live   # 로컬 모델 서버가 필요합니다
```

모델 서버 주소와 모델 이름은 `AGENT_BASE_URL`, `AGENT_MODEL` 환경 변수로 바꿉니다. OpenAI 호환 서버라면 어느 것이든 됩니다.

S4 문서 서버를 쓰는 LangChain 에이전트는 `cd s4-mcp && uv run python -m docs_mcp.agent "질문"`으로 실행합니다(로컬 모델 서버 필요).

`.mcp.json`은 S4 문서 서버(`s4-mcp/`)를 Claude Code에 프로젝트 범위로 연결합니다. 저장소 루트에서 `claude`를 실행하면 처음 한 번 승인 창이 뜹니다.
