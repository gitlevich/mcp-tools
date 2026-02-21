import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config as cfg
from indexer import ThoughtIndex, SearchResult
from sources import Source, save_sources


def _make_index(tmp_path: Path, sources: list[Source]) -> ThoughtIndex:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    chroma_dir = tmp_path / "chroma"

    save_sources(sources)

    return ThoughtIndex(chroma_dir)


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(cfg, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg, "SOURCES_FILE", config_dir / "sources.json")


@pytest.fixture()
def two_sources(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    journal = tmp_path / "journal"
    journal.mkdir()

    sources = [
        Source(path=str(notes), label="notes"),
        Source(path=str(journal), label="journal"),
    ]
    index = _make_index(tmp_path, sources)
    return index, notes, journal


def test_scan_files_across_sources(two_sources):
    index, notes, journal = two_sources

    (notes / "idea.md").write_text("# Great idea")
    (journal / "entry.txt").write_text("Today was good.")

    files = index.scan_files()
    names = {p.name for p, _ in files}
    assert names == {"idea.md", "entry.txt"}


def test_scan_files_skips_missing_source(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    sources = [
        Source(path=str(existing), label="existing"),
        Source(path=str(tmp_path / "gone"), label="gone"),
    ]
    index = _make_index(tmp_path, sources)
    (existing / "note.md").write_text("# Hello")

    files = index.scan_files()
    assert len(files) == 1


def test_index_and_search(two_sources):
    index, notes, journal = two_sources

    (notes / "cooking.md").write_text("# Recipe\n\nMake a delicious pasta.")
    (journal / "monday.txt").write_text("Went for a long walk in the park.")

    index._do_refresh()
    assert index._collection.count() >= 2

    results = index.search("pasta recipe")
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


def test_refresh_detects_changes(two_sources):
    index, notes, _ = two_sources

    f = notes / "evolving.md"
    f.write_text("# Version 1")
    index._do_refresh()
    count_before = index._collection.count()

    time.sleep(0.05)
    f.write_text("# Version 2\n\nNew content added.")

    index._last_scan = None  # force refresh
    index._do_refresh()
    assert index._collection.count() >= count_before


def test_refresh_removes_deleted(two_sources):
    index, notes, _ = two_sources

    f = notes / "temp.md"
    f.write_text("# Temporary")
    index._do_refresh()
    assert index._collection.count() > 0

    f.unlink()
    index._last_scan = None
    index._do_refresh()
    assert index._collection.count() == 0


def test_remove_source_chunks(two_sources):
    index, notes, journal = two_sources

    (notes / "a.md").write_text("# Note A")
    (journal / "b.txt").write_text("Journal B")
    index._do_refresh()

    before = index._collection.count()
    removed = index.remove_source_chunks("notes")

    assert removed > 0
    assert index._collection.count() == before - removed


def test_full_reindex(two_sources):
    index, notes, journal = two_sources

    (notes / "x.md").write_text("# X")
    (journal / "y.txt").write_text("Y content")

    stats = index.full_reindex()
    assert stats["files"] == 2
    assert stats["chunks"] >= 2
    assert stats["sources"] == 2
    assert stats["elapsed"] >= 0


def test_status_empty(tmp_path):
    index = _make_index(tmp_path, [])
    s = index.status()
    assert s["total_files"] == 0
    assert s["total_chunks"] == 0
    assert s["last_scan"] is None


def test_status_after_index(two_sources):
    index, notes, journal = two_sources

    (notes / "one.md").write_text("# One")
    (journal / "two.txt").write_text("Two")
    index._do_refresh()

    s = index.status()
    assert s["total_files"] == 2
    assert s["total_chunks"] >= 2
    assert s["pending_reindex"] == 0
    assert s["source_count"] == 2


def test_source_file_count(two_sources):
    index, notes, journal = two_sources

    (notes / "a.md").write_text("# A")
    (notes / "b.md").write_text("# B")
    (journal / "c.txt").write_text("C")
    index._do_refresh()

    assert index.source_file_count("notes") == 2
    assert index.source_file_count("journal") == 1


def test_index_source(two_sources):
    index, notes, _ = two_sources

    (notes / "fresh.md").write_text("# Fresh content")
    source = Source(path=str(notes), label="notes")

    stats = index.index_source(source)
    assert stats["files"] == 1
    assert stats["chunks"] >= 1


def test_search_diversity(two_sources, monkeypatch):
    """Search results should be capped per file, surfacing multiple files."""
    import config as test_cfg

    monkeypatch.setattr(test_cfg, "MAX_PER_FILE", 1)

    index, notes, journal = two_sources

    # Create files with similar content so they'd all match the same query.
    for i in range(5):
        (notes / f"recipe_{i}.md").write_text(
            f"# Recipe {i}\n\nMake a delicious pasta with tomato sauce."
        )
    for i in range(5):
        (journal / f"cooking_{i}.txt").write_text(
            f"Today I cooked a wonderful pasta dish with fresh tomatoes."
        )

    index._do_refresh()
    results = index.search("pasta recipe", top_k=6)

    assert len(results) == 6
    files = [r.rel_path for r in results]
    # With MAX_PER_FILE=1, all 6 results must come from different files.
    assert len(set(files)) == 6
