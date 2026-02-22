import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexer import CodeIndex, SearchResult, rrf


def _mock_openai_embeddings(texts: list[str]) -> list[list[float]]:
    """Return deterministic fake embeddings (1536-dim)."""
    results = []
    for i, text in enumerate(texts):
        # Simple hash-based embedding for deterministic results
        base = hash(text) % 1000 / 1000.0
        vec = [base + (j * 0.001) for j in range(1536)]
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        results.append([v / norm for v in vec])
    return results


class FakeEmbeddingData:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [FakeEmbeddingData(e) for e in embeddings]


def _make_index(tmp_path: Path) -> CodeIndex:
    """Create a CodeIndex with mocked OpenAI client."""
    project = tmp_path / "project"
    project.mkdir()
    chroma = tmp_path / "chroma"

    index = CodeIndex(project, chroma)

    # Mock OpenAI embeddings
    def fake_create(model, input):
        embeddings = _mock_openai_embeddings(input)
        return FakeEmbeddingResponse(embeddings)

    index._openai = MagicMock()
    index._openai.embeddings.create = fake_create

    return index


def test_scan_files_finds_python(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    (project / "app.py").write_text("print('hi')")
    (project / "readme.md").write_text("# Hello")
    (project / "data.json").write_text("{}")

    files = index.scan_files()
    suffixes = {f.suffix for f in files}

    assert ".py" in suffixes
    assert ".md" in suffixes
    assert ".json" not in suffixes


def test_scan_files_excludes_venv(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    venv = project / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "something.py").write_text("x = 1")
    (project / "app.py").write_text("y = 2")

    files = index.scan_files()

    assert len(files) == 1
    assert files[0].name == "app.py"


def test_index_and_search(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    (project / "taste.py").write_text("def learn_taste():\n    pass\n")
    (project / "projection.py").write_text("def score_category():\n    pass\n")

    stats = index.refresh()

    assert stats["reindexed_files"] == 2
    assert stats["total_chunks"] >= 2

    results = index.search("taste learning")

    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.start_line >= 1 for r in results)


def test_refresh_detects_changes(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    f = project / "module.py"
    f.write_text("v1 = True")
    index.refresh()

    count_before = index._collection.count()

    # Modify file
    time.sleep(0.05)  # ensure mtime changes
    f.write_text("v2 = True\n\ndef new_func():\n    pass\n")

    stats = index.refresh()

    assert stats["reindexed_files"] == 1


def test_refresh_removes_deleted(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    f = project / "temp.py"
    f.write_text("x = 1")
    index.refresh()

    assert index._collection.count() > 0

    f.unlink()
    stats = index.refresh()

    assert stats["removed_files"] == 1
    assert index._collection.count() == 0


def test_full_reindex(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    (project / "a.py").write_text("a = 1")
    (project / "b.py").write_text("b = 2")

    stats = index.full_reindex()

    assert stats["files"] == 2
    assert stats["chunks"] >= 2
    assert stats["elapsed"] >= 0


def test_status_empty(tmp_path: Path):
    index = _make_index(tmp_path)

    s = index.status()

    assert s["total_files"] == 0
    assert s["total_chunks"] == 0
    assert s["last_scan"] is None


def test_status_after_index(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    (project / "one.py").write_text("x = 1")
    (project / "two.py").write_text("y = 2")
    index.refresh()

    s = index.status()

    assert s["total_files"] == 2
    assert s["total_chunks"] >= 2
    assert s["pending_reindex"] == 0


# -- RRF unit tests --


def test_rrf_single_source():
    rankings = {"vector": ["a", "b", "c"]}
    result = rrf(rankings, {"vector": 1.0}, top_k=3)
    ids = [cid for cid, _ in result]
    assert ids == ["a", "b", "c"]


def test_rrf_keyword_weight_promotes():
    """Higher-weighted keyword source should promote its top result."""
    rankings = {
        "vector": ["a", "b", "c"],
        "keyword": ["c", "b"],
    }
    result = rrf(rankings, {"vector": 1.0, "keyword": 3.0}, top_k=3)
    ids = [cid for cid, _ in result]
    assert ids[0] == "c"


def test_rrf_empty():
    assert rrf({}, {}, top_k=5) == []


# -- FTS sync tests --


def test_fts_populated_after_index(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    (project / "cities.py").write_text("Vancouver = 'a city in BC'")
    index.refresh()

    hits = index._fts_search("Vancouver", limit=5)
    assert len(hits) > 0


def test_fts_cleared_after_delete(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    f = project / "temp.py"
    f.write_text("Vancouver = 'test'")
    index.refresh()

    assert len(index._fts_search("Vancouver", limit=5)) > 0

    f.unlink()
    index.refresh()

    assert len(index._fts_search("Vancouver", limit=5)) == 0


def test_fts_rebuilt_after_full_reindex(tmp_path: Path):
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    (project / "a.py").write_text("Vancouver = 1")
    index.refresh()

    (project / "a.py").write_text("Toronto = 2")
    index.full_reindex()

    assert len(index._fts_search("Vancouver", limit=5)) == 0
    assert len(index._fts_search("Toronto", limit=5)) > 0


# -- Hybrid search integration --


def test_hybrid_search_boosts_keyword_match(tmp_path: Path):
    """Chunk containing the literal query term should rank first."""
    index = _make_index(tmp_path)
    project = tmp_path / "project"

    (project / "cities.py").write_text(
        "# Cities\nVancouver = 'a beautiful city in British Columbia'\n"
    )
    (project / "nature.py").write_text(
        "# Nature\nbeautiful_mountains = 'scenic views and forests'\n"
    )
    (project / "weather.py").write_text(
        "# Weather\nrain = 'common in Pacific Northwest coastal areas'\n"
    )

    index.refresh()
    results = index.search("Vancouver")

    assert len(results) > 0
    assert "Vancouver" in results[0].text
