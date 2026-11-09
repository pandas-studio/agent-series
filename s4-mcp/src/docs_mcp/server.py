"""An MCP server over local documentation pages (Markdown/MDX files).

Run over stdio: uv run python -m docs_mcp.server
The doc folder is DOCS_ROOT (default: ../corpus next to this project).
"""

import os
import re
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "corpus"
SUFFIXES = {".md", ".mdx"}
WORD = re.compile(r"[\w-]+", re.UNICODE)

mcp = FastMCP("docs", instructions="Search and read local LangGraph/LangChain documentation pages.")


def docs_root() -> Path:
    return Path(os.environ.get("DOCS_ROOT", DEFAULT_ROOT)).resolve()


def _pages(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix in SUFFIXES and p.is_file())


def _title(text: str, fallback: str) -> str:
    m = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", text, re.M) or re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else fallback


def _resolve(path: str) -> Path:
    """Map a page path from search_docs to a file, refusing anything outside the doc folder."""
    root = docs_root()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or target.suffix not in SUFFIXES or not target.is_file():
        # ToolError: the client gets this message as a tool error; the server logs no traceback.
        raise ToolError(f"unknown page: {path}")
    return target


@mcp.tool
def search_docs(query: str, limit: int = 5) -> list[dict]:
    """Find documentation pages about a topic.

    Returns the best pages first, each with its path, title, number of matched words and one matching line.
    Use read_doc with a returned path to read the page.
    """
    words = {w.lower() for w in WORD.findall(query) if len(w) > 1}
    if not words:
        return []
    root = docs_root()
    hits = []
    for page in _pages(root):
        text = page.read_text(encoding="utf-8", errors="replace")
        lower = text.lower()
        score = sum(lower.count(w) for w in words)
        if not score:
            continue
        line = next((ln.strip() for ln in text.splitlines() if any(w in ln.lower() for w in words)), "")
        hits.append({"path": page.relative_to(root).as_posix(), "title": _title(text, page.stem),
                     "score": score, "line": line[:200]})
    hits.sort(key=lambda h: (-h["score"], h["path"]))
    return hits[:max(1, min(limit, 20))]


@mcp.tool
def read_doc(path: str, start: int = 0, max_chars: int = 4000) -> dict:
    """Read part of a documentation page.

    path comes from search_docs. Pages can be long: read max_chars characters from start, and continue with
    start=next_start while more is true.
    """
    text = _resolve(path).read_text(encoding="utf-8", errors="replace")
    max_chars = max(200, min(max_chars, 20000))
    start = max(0, start)
    end = min(len(text), start + max_chars)
    return {"path": path, "start": start, "next_start": end, "more": end < len(text), "total_chars": len(text),
            "text": text[start:end]}


if __name__ == "__main__":
    # stdout carries the protocol over stdio; the banner would only add noise to the client's stderr log.
    mcp.run(show_banner=False)
