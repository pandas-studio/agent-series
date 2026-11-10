"""An MCP server over local documentation pages (Markdown/MDX files).

Three MCP primitives, each with a different owner (MCP spec, server features):
- tools (search_docs, read_doc): the model decides when to call them;
- resources (docs://index, docs://page/{path*}): the client application decides what to attach;
- prompts (ask_docs, summarize_page): the user picks one, like a slash command.

Run over stdio: uv run python -m docs_mcp.server
The doc folder is DOCS_ROOT (default: ../corpus next to this project).
"""

import json
import os
import re
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import FastMCPError, PromptError, ResourceError, ToolError
from fastmcp.prompts import Message
from mcp.types import EmbeddedResource, TextResourceContents

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


def _resolve(path: str, error: type[FastMCPError] = ToolError) -> Path:
    """Map a page path to a file, refusing anything outside the doc folder.

    error is the FastMCP error class of the calling primitive. The client gets its message either way; for a tool
    the server logs one line, for a resource or a prompt FastMCP 4.1.6 also logs the traceback.
    """
    root = docs_root()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or target.suffix not in SUFFIXES or not target.is_file():
        raise error(f"unknown page: {path}")
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


@mcp.resource("docs://index", mime_type="application/json")
def index() -> str:
    """Every doc page: its path, title, size in characters and resource URI."""
    root = docs_root()
    pages = []
    for page in _pages(root):
        text = page.read_text(encoding="utf-8", errors="replace")
        path = page.relative_to(root).as_posix()
        pages.append({"path": path, "title": _title(text, page.stem), "chars": len(text),
                      "uri": f"docs://page/{path}"})
    return json.dumps(pages, ensure_ascii=False)


# {path*} matches several URI segments (langgraph/interrupts.mdx); a plain {path} matches one segment only.
@mcp.resource("docs://page/{path*}", mime_type="text/markdown")
def page(path: str) -> str:
    """The whole text of one doc page. Use docs://index for the paths."""
    return _resolve(path, ResourceError).read_text(encoding="utf-8", errors="replace")


@mcp.prompt
def ask_docs(question: str) -> str:
    """Answer a question from the local docs with search_docs and read_doc, naming the pages used."""
    return (f"Question: {question}\n\n"
            "Answer only from the local documentation. First call search_docs with the key English terms of the "
            "question, then read the most relevant page with read_doc (continue with next_start if the answer is "
            "further down). Answer in Korean and list the page paths you used. If the pages do not answer the "
            "question, say so.")


@mcp.prompt
def summarize_page(path: str) -> list[Message]:
    """Summarize one doc page; the page is embedded in the prompt as a resource."""
    # A plain dict here is refused (TypeError), and Message(dict) would turn the resource into JSON text:
    # the embedded resource has to be an EmbeddedResource.
    text = _resolve(path, PromptError).read_text(encoding="utf-8", errors="replace")
    resource = TextResourceContents(uri=f"docs://page/{path}", mime_type="text/markdown", text=text)
    return [
        Message(EmbeddedResource(type="resource", resource=resource)),
        Message("Summarize this page in Korean in 5 bullet points. Keep API names as written."),
    ]


if __name__ == "__main__":
    # stdout carries the protocol over stdio; the banner would only add noise to the client's stderr log.
    mcp.run(show_banner=False)
