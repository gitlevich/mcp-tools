import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from entities import EntityStore


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test_entities.db"
    s = EntityStore(db)
    yield s
    s.close()


def _make_chunks(texts, rel_paths=None, mtimes=None):
    """Helper to create documents + metadatas lists."""
    documents = texts
    metadatas = []
    for i, text in enumerate(texts):
        metadatas.append({
            "rel_path": (rel_paths[i] if rel_paths else f"notes/doc{i}.md"),
            "chunk_index": 0,
            "mtime": (mtimes[i] if mtimes else 1000.0),
            "file_path": f"/tmp/doc{i}.md",
            "start_line": 1,
            "end_line": 10,
        })
    return documents, metadatas


class TestExtraction:
    def test_basic_extraction(self, store):
        docs, metas = _make_chunks([
            "Eric Schmidt met with Sundar Pichai at Google headquarters in Mountain View."
        ])
        stats = store.extract_from_chunks(docs, metas)
        assert stats["chunks_processed"] == 1
        assert stats["mentions_added"] > 0

    def test_incremental_skips_processed(self, store):
        docs, metas = _make_chunks(["Eric Schmidt works at Google."])
        store.extract_from_chunks(docs, metas)

        stats = store.extract_from_chunks(docs, metas)
        assert stats["chunks_skipped"] == 1
        assert stats["chunks_processed"] == 0

    def test_reextracts_on_mtime_change(self, store):
        docs, metas = _make_chunks(["Eric Schmidt works at Google."], mtimes=[1000.0])
        store.extract_from_chunks(docs, metas)
        first_mentions = store.stats()["total_mentions"]

        docs2, metas2 = _make_chunks(
            ["Eric Schmidt and Sundar Pichai work at Google and Alphabet."],
            mtimes=[2000.0],
        )
        store.extract_from_chunks(docs2, metas2)
        assert store.stats()["total_mentions"] >= first_mentions


class TestSearch:
    def test_search_by_name(self, store):
        docs, metas = _make_chunks([
            "Eric Schmidt was CEO. Tim Cook is CEO of Apple."
        ])
        store.extract_from_chunks(docs, metas)

        results = store.search("Eric")
        assert len(results) >= 1
        assert any("Eric" in e.canonical_name for e in results)

    def test_search_by_type(self, store):
        docs, metas = _make_chunks([
            "Barack Obama visited Paris and met with representatives from Google."
        ])
        store.extract_from_chunks(docs, metas)

        people = store.search("", entity_type="person")
        for e in people:
            assert e.type == "person"

    def test_search_finds_alias(self, store):
        docs, metas = _make_chunks(["Eric Schmidt works at Google."])
        store.extract_from_chunks(docs, metas)

        results = store.search("Eric")
        assert len(results) >= 1
        entity_id = results[0].id

        store.rename(entity_id, "E. Schmidt")
        results = store.search("Eric")
        assert len(results) >= 1


class TestMerge:
    def test_merge_combines_mentions(self, store):
        docs, metas = _make_chunks(
            [
                "Eric Schmidt spoke at the event.",
                "E. Schmidt published a paper.",
            ],
            rel_paths=["notes/a.md", "notes/b.md"],
        )
        store.extract_from_chunks(docs, metas)

        erics = store.search("Schmidt")
        if len(erics) >= 2:
            result = store.merge(erics[0].id, erics[1].id)
            assert result is not None
            assert result.mention_count == erics[0].mention_count + erics[1].mention_count

    def test_merge_adds_old_name_as_alias(self, store):
        docs, metas = _make_chunks(
            ["Alice works at Acme.", "Bob works at Acme Corp."],
            rel_paths=["notes/a.md", "notes/b.md"],
        )
        store.extract_from_chunks(docs, metas)

        acmes = [e for e in store.search("Acme") if e.type == "org"]
        if len(acmes) >= 2:
            result = store.merge(acmes[0].id, acmes[1].id)
            assert result.aliases is not None
            assert len(result.aliases) >= 1


class TestRename:
    def test_rename_preserves_old_as_alias(self, store):
        docs, metas = _make_chunks(["Eric Schmidt works at Google."])
        store.extract_from_chunks(docs, metas)

        results = store.search("Eric Schmidt")
        assert len(results) >= 1
        entity = results[0]

        renamed = store.rename(entity.id, "E. Schmidt")
        assert renamed.canonical_name == "E. Schmidt"
        assert "Eric Schmidt" in renamed.aliases


class TestListAndStats:
    def test_list_entities_sorted_by_mentions(self, store):
        docs, metas = _make_chunks(
            [
                "Eric Schmidt and Google and Eric Schmidt again.",
                "Tim Cook at Apple.",
            ],
            rel_paths=["notes/a.md", "notes/b.md"],
        )
        store.extract_from_chunks(docs, metas)

        entities = store.list_entities()
        assert len(entities) > 0
        for i in range(len(entities) - 1):
            assert entities[i].mention_count >= entities[i + 1].mention_count

    def test_stats_returns_counts(self, store):
        docs, metas = _make_chunks([
            "Barack Obama visited Paris and met with representatives from Google."
        ])
        store.extract_from_chunks(docs, metas)

        s = store.stats()
        assert s["total_entities"] > 0
        assert s["total_mentions"] > 0
        assert isinstance(s["by_type"], dict)


class TestDelete:
    def test_delete_entity(self, store):
        docs, metas = _make_chunks(["Eric Schmidt works at Google."])
        store.extract_from_chunks(docs, metas)

        results = store.search("Eric")
        assert len(results) >= 1
        entity_id = results[0].id

        assert store.delete(entity_id)
        assert store.get(entity_id) is None

    def test_delete_nonexistent(self, store):
        assert not store.delete(99999)


class TestMentions:
    def test_get_mentions_with_context(self, store):
        docs, metas = _make_chunks(["Eric Schmidt was the CEO of Google."])
        store.extract_from_chunks(docs, metas)

        results = store.search("Eric")
        assert len(results) >= 1

        mentions = store.mentions(results[0].id)
        assert len(mentions) >= 1
        assert mentions[0].context is not None
        assert mentions[0].rel_path == "notes/doc0.md"
