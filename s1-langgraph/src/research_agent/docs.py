"""Plain Python search over the local doc pages (../corpus). No model is involved.

The S4 series turns the same idea into an MCP server; here it is a function that a graph node calls.
"""

import math
import os
import re
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "corpus"
SUFFIXES = {".md", ".mdx"}
# ASCII only: with \w, a Korean particle stuck to the term ("checkpointer는") became part of the word and nothing matched.
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


def docs_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("DOCS_ROOT", DEFAULT_ROOT)).resolve()


def _title(text: str, fallback: str) -> str:
    m = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", text, re.M) or re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else fallback


def search(query: str, root: str | Path | None = None, limit: int = 3, chars: int = 1500) -> list[dict]:
    """Pages that mention the query's rarer English words most, best first: path, title, score and an excerpt.

    Only English words count (API and product names such as checkpointer or StateGraph), because the pages are in
    English and the questions are usually Korean with English terms.
    """
    words = {w.lower() for w in WORD.findall(query)}
    if not words:
        return []
    base = docs_root(root)
    pages = [(p, p.read_text(encoding="utf-8", errors="replace"))
             for p in sorted(p for p in base.rglob("*") if p.suffix in SUFFIXES and p.is_file())]
    # A word on every page tells pages apart by nothing: "langgraph" is on all 16 pages, 229 times in
    # graph-api-control-flow.mdx, and plain counts ranked that page above checkpointers.mdx for a checkpointer question.
    df = {w: sum(w in text.lower() for _, text in pages) for w in words}
    weight = {w: math.log((len(pages) + 1) / (df[w] + 1)) for w in words}
    hits = []
    for page, text in pages:
        lower = text.lower()
        score = round(sum(lower.count(w) * weight[w] for w in words), 2)
        if score <= 0:
            continue
        first = min((lower.find(w) for w in words if w in lower), default=0)
        start = max(0, text.rfind("\n", 0, max(0, first - chars // 3)) + 1)
        hits.append({"path": page.relative_to(base).as_posix(), "title": _title(text, page.stem), "score": score,
                     "excerpt": text[start:start + chars]})
    hits.sort(key=lambda h: (-h["score"], h["path"]))
    return hits[:limit]
