"""Episode 3: the doc search as a tool the model calls.

In episode 2 a graph node searched with the user's question as is. Here the model decides whether to search and
writes the query itself. The docstring is the tool description the model reads; it says the pages are in English.
tool_check.py also runs the same tool with a vague description (VAGUE_DOC) to see what that sentence changes.
"""

from langchain_core.tools import tool

from .docs import search

VAGUE_DOC = "Search the docs."


@tool
def search_docs(query: str) -> str:
    """Search the local LangGraph and LangChain documentation pages.

    The pages are in English: write the query as English keywords and API names (for example "conditional edges
    StateGraph", not a Korean sentence). Returns the best pages with their path and an excerpt.
    """
    hits = search(query, limit=3, chars=1200)   # the doc folder comes from DOCS_ROOT (tests point it elsewhere)
    if not hits:
        return f"No page matched {query!r}."
    return "\n\n".join(f"[{h['path']}] {h['title']}\n{h['excerpt']}" for h in hits)


def with_description(text: str):
    """The same tool with another description (the schema and the function stay the same)."""
    return search_docs.model_copy(update={"description": text})
