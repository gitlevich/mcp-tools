import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity import ActivityReporter

from config import DEFAULT_TOP_K, CHROMA_DIR
from indexer import ThoughtIndex
from picker import pick_folder
from sources import load_sources, add_source, remove_source

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("thought-index")
reporter = ActivityReporter("thought-index")

_index: ThoughtIndex | None = None


def get_index() -> ThoughtIndex:
    global _index
    if _index is None:
        _index = ThoughtIndex(CHROMA_DIR)
    return _index


@mcp.tool()
def search_thoughts(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Search across all indexed thought folders by semantic similarity.

    Takes a natural language query and returns the most relevant chunks
    from notes and documents, with file paths, line numbers, and content.

    Args:
        query: Natural language description of what to find.
        top_k: Number of results to return (default 10).
    """
    with reporter.report("Searching thoughts") as r:
        index = get_index()
        index.schedule_refresh(progress_callback=r.set_progress)

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
def add_source_folder(path: str = "") -> str:
    """Add a folder of notes to the index.

    The folder will be tracked and its contents included in future searches.
    Opens a native file dialog if no path is provided.

    Args:
        path: Absolute path to the folder. If empty, opens a Finder dialog.
    """
    with reporter.report("Adding source folder") as r:
        folder = path.strip() or pick_folder()
        if not folder:
            return "No folder selected."

        try:
            source = add_source(folder)
        except ValueError as e:
            return str(e)

        index = get_index()
        stats = index.index_source(source, progress_callback=r.set_progress)

        return (
            f"Added source '{source.label}' ({source.path}). "
            f"Indexed {stats['files']} files, {stats['chunks']} chunks."
        )


@mcp.tool()
def remove_source_folder(label: str) -> str:
    """Remove a folder from the index by its label.

    Stops tracking the folder and removes all its chunks from the index.

    Args:
        label: The label of the source to remove (see list_sources).
    """
    with reporter.report("Removing source folder"):
        try:
            source = remove_source(label)
        except ValueError as e:
            return str(e)

        index = get_index()
        removed = index.remove_source_chunks(label)
        return f"Removed source '{label}' ({source.path}). Deleted {removed} chunks."


@mcp.tool()
def list_sources() -> str:
    """List all configured source folders.

    Shows each folder's label, path, and indexed file count.
    """
    sources = load_sources()
    if not sources:
        return "No sources configured. Use add_source_folder to add one."

    index = get_index()
    lines = []
    for s in sources:
        count = index.source_file_count(s.label)
        lines.append(f"  {s.label}: {s.path} ({count} files)")
    return "Sources:\n" + "\n".join(lines)


@mcp.tool()
def reindex() -> str:
    """Force a full re-index of all source folders.

    Drops the existing index and rebuilds from scratch.
    Use when the index seems stale or corrupted.
    """
    with reporter.report("Full reindex") as r:
        index = get_index()
        stats = index.full_reindex(progress_callback=r.set_progress)
        return (
            f"Re-indexed {stats['files']} files into {stats['chunks']} chunks "
            f"from {stats['sources']} sources in {stats['elapsed']:.1f}s."
        )


if __name__ == "__main__":
    import atexit
    atexit.register(reporter.cleanup)
    mcp.run(transport="stdio")
