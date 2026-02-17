import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from config import DEFAULT_TOP_K, PROJECT_ROOT, CHROMA_DIR
from indexer import CodeIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("project-embed")

_index: CodeIndex | None = None


def get_index() -> CodeIndex:
    global _index
    if _index is None:
        _index = CodeIndex(PROJECT_ROOT, CHROMA_DIR)
    return _index


@mcp.tool()
def search(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Search the project by semantic similarity.

    Takes a natural language query and returns the most relevant chunks
    from code and documentation, with file paths, line numbers, and content.

    Args:
        query: Natural language description of what to find.
        top_k: Number of results to return (default 10).
    """
    index = get_index()
    stats = index.refresh()

    if stats["reindexed_files"] > 0:
        logger.info(
            "Refreshed index: %d files re-indexed, %d chunks",
            stats["reindexed_files"],
            stats["reindexed_chunks"],
        )

    results = index.search(query, top_k=top_k)

    if not results:
        return "No results found."

    parts = []
    for r in results:
        parts.append(
            f"--- {r.rel_path} (lines {r.start_line}-{r.end_line}, "
            f"score: {r.score:.3f}) ---\n{r.text}"
        )
    return "\n\n".join(parts)


@mcp.tool()
def index_status() -> str:
    """Report the current state of the search index.

    Returns total files indexed, total chunks, last scan time,
    and count of files pending re-index.
    """
    index = get_index()
    s = index.status()
    lines = [
        f"Total files indexed: {s['total_files']}",
        f"Total chunks: {s['total_chunks']}",
        f"Last scan: {s['last_scan'] or 'never'}",
        f"Files pending re-index: {s['pending_reindex']}",
    ]
    return "\n".join(lines)


@mcp.tool()
def reindex() -> str:
    """Force a full re-index of the project.

    Drops the existing index and rebuilds from scratch.
    Use when the index seems stale or corrupted.
    """
    index = get_index()
    stats = index.full_reindex()
    return (
        f"Re-indexed {stats['files']} files into {stats['chunks']} chunks "
        f"in {stats['elapsed']:.1f}s."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
