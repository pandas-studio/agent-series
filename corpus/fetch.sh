#!/bin/sh
# Download a pinned set of LangGraph/LangChain doc pages (langchain-ai/docs, MIT) for the examples.
# The pages are not committed; rerun this script to refresh them.
set -eu
cd "$(dirname "$0")"
REF="${DOCS_REF:-main}"
BASE="https://raw.githubusercontent.com/langchain-ai/docs/$REF/src/oss"
mkdir -p langgraph langchain
for p in overview quickstart graph-api graph-api-state graph-api-nodes graph-api-control-flow \
         persistence checkpointers interrupts streaming use-subgraphs functional-api; do
  curl -fsSL "$BASE/langgraph/$p.mdx" -o "langgraph/$p.mdx"
done
for p in agents tools short-term-memory human-in-the-loop; do
  curl -fsSL "$BASE/langchain/$p.mdx" -o "langchain/$p.mdx" || echo "skip langchain/$p"
done
git ls-remote https://github.com/langchain-ai/docs "$REF" | cut -c1-12 > SOURCE_COMMIT
echo "fetched $(ls langgraph langchain | grep -c mdx) pages at $(cat SOURCE_COMMIT)"
