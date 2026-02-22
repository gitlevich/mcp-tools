import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity import ActivityReporter

from config import DEFAULT_TOP_K, CHROMA_DIR, ENTITIES_DB
from entities import EntityStore
from indexer import ThoughtIndex
from picker import pick_folder
from sources import load_sources, add_source, remove_source

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("thought-index")
reporter = ActivityReporter("thought-index")

_index: ThoughtIndex | None = None
_entity_store: EntityStore | None = None


def get_index() -> ThoughtIndex:
    global _index
    if _index is None:
        _index = ThoughtIndex(CHROMA_DIR)
    return _index


def get_entity_store() -> EntityStore:
    global _entity_store
    if _entity_store is None:
        _entity_store = EntityStore(ENTITIES_DB)
    return _entity_store


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


# -- Entity resolution tools --


@mcp.tool()
def extract_entities() -> str:
    """Run named entity recognition on all indexed note chunks.

    Extracts people, places, and organizations using spaCy NER.
    Incremental: skips chunks already processed since their last modification.
    """
    with reporter.report("Extracting entities") as r:
        index = get_index()
        if index._collection.count() == 0:
            return "No chunks indexed. Run a search or reindex first."

        all_data = index._get_all(include=["documents", "metadatas"])
        documents = all_data["documents"]
        metadatas = all_data["metadatas"]

        store = get_entity_store()
        stats = store.extract_from_chunks(
            documents, metadatas, progress_callback=r.set_progress,
        )
        return (
            f"Processed {stats['chunks_processed']} chunks "
            f"(skipped {stats['chunks_skipped']}). "
            f"Added {stats['mentions_added']} entity mentions."
        )


@mcp.tool()
def search_entities(query: str, type: str = "") -> str:
    """Find entities by name substring.

    Searches canonical names and aliases. Returns matches with mention counts.

    Args:
        query: Name or partial name to search for (e.g. "Eric").
        type: Optional filter: "person", "place", or "org".
    """
    store = get_entity_store()
    entity_type = type.strip() or None
    results = store.search(query, entity_type=entity_type)

    if not results:
        return f"No entities matching '{query}'."

    lines = []
    for e in results:
        aliases = ""
        full = store.get(e.id)
        if full and full.aliases:
            aliases = f" (aka: {', '.join(full.aliases)})"
        lines.append(f"  [{e.id}] {e.canonical_name} ({e.type}) - {e.mention_count} mentions{aliases}")
    return "Entities:\n" + "\n".join(lines)


@mcp.tool()
def get_entity(entity_id: int) -> str:
    """Get detailed information about an entity.

    Shows canonical name, type, aliases, and sample mentions with context.

    Args:
        entity_id: The entity ID (from search_entities or list_entities).
    """
    store = get_entity_store()
    entity = store.get(entity_id)
    if not entity:
        return f"Entity {entity_id} not found."

    parts = [f"{entity.canonical_name} ({entity.type}) - {entity.mention_count} mentions"]
    if entity.aliases:
        parts.append(f"Aliases: {', '.join(entity.aliases)}")

    sample_mentions = store.mentions(entity_id, limit=10)
    if sample_mentions:
        parts.append("\nMentions:")
        for m in sample_mentions:
            loc = f"{m.rel_path}"
            if m.start_line:
                loc += f":{m.start_line}"
            ctx = f' - "{m.context}"' if m.context else ""
            parts.append(f"  {loc}{ctx}")

    return "\n".join(parts)


@mcp.tool()
def entity_mentions(entity_id: int, limit: int = 50) -> str:
    """List all mentions of an entity across notes.

    Args:
        entity_id: The entity ID.
        limit: Maximum mentions to return (default 50).
    """
    store = get_entity_store()
    entity = store.get(entity_id)
    if not entity:
        return f"Entity {entity_id} not found."

    all_mentions = store.mentions(entity_id, limit=limit)
    if not all_mentions:
        return f"No mentions found for {entity.canonical_name}."

    parts = [f"Mentions of {entity.canonical_name} ({entity.type}):"]
    for m in all_mentions:
        loc = m.rel_path
        if m.start_line:
            loc += f":{m.start_line}-{m.end_line}"
        ctx = f'\n    "{m.context}"' if m.context else ""
        parts.append(f"  [{m.id}] {loc}{ctx}")
    return "\n".join(parts)


@mcp.tool()
def merge_entities(keep_id: int, merge_id: int) -> str:
    """Merge two entities that refer to the same real-world thing.

    All mentions and aliases from merge_id are moved to keep_id.
    The merged entity is deleted.

    Args:
        keep_id: Entity ID to keep as the canonical entity.
        merge_id: Entity ID to merge into the kept entity.
    """
    store = get_entity_store()
    keep = store.get(keep_id)
    merge = store.get(merge_id)

    if not keep:
        return f"Entity {keep_id} not found."
    if not merge:
        return f"Entity {merge_id} not found."

    result = store.merge(keep_id, merge_id)
    if not result:
        return "Merge failed."

    aliases = f" (aliases: {', '.join(result.aliases)})" if result.aliases else ""
    return (
        f"Merged '{merge.canonical_name}' into '{result.canonical_name}'. "
        f"Now has {result.mention_count} mentions{aliases}."
    )


@mcp.tool()
def rename_entity(entity_id: int, new_name: str) -> str:
    """Rename an entity's canonical name. The old name becomes an alias.

    Args:
        entity_id: The entity ID to rename.
        new_name: The new canonical name.
    """
    store = get_entity_store()
    result = store.rename(entity_id, new_name)
    if not result:
        return f"Entity {entity_id} not found."

    aliases = f" (aliases: {', '.join(result.aliases)})" if result.aliases else ""
    return f"Renamed to '{result.canonical_name}'{aliases}."


@mcp.tool()
def list_entities(type: str = "", limit: int = 50) -> str:
    """Browse all extracted entities, sorted by mention count.

    Args:
        type: Optional filter: "person", "place", or "org".
        limit: Maximum results (default 50).
    """
    store = get_entity_store()
    entity_type = type.strip() or None
    results = store.list_entities(entity_type=entity_type, limit=limit)

    if not results:
        return "No entities found. Run extract_entities first."

    lines = []
    for e in results:
        lines.append(f"  [{e.id}] {e.canonical_name} ({e.type}) - {e.mention_count} mentions")
    return "Entities:\n" + "\n".join(lines)


@mcp.tool()
def entity_stats() -> str:
    """Summary of extracted entities: counts by type and total mentions."""
    store = get_entity_store()
    s = store.stats()

    parts = [f"Total entities: {s['total_entities']}, Total mentions: {s['total_mentions']}"]
    for t, count in sorted(s["by_type"].items()):
        parts.append(f"  {t}: {count}")
    return "\n".join(parts)


if __name__ == "__main__":
    import atexit
    atexit.register(reporter.cleanup)
    mcp.run(transport="stdio")
