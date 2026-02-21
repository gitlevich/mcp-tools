import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config as cfg
from sources import Source, load_sources, save_sources, add_source, remove_source, _unique_label


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SOURCES_FILE", tmp_path / "sources.json")


def test_load_empty_returns_empty_list():
    assert load_sources() == []


def test_add_source_creates_file(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    source = add_source(str(folder))

    assert source.path == str(folder)
    assert source.label == "notes"
    assert cfg.SOURCES_FILE.exists()

    data = json.loads(cfg.SOURCES_FILE.read_text())
    assert len(data["sources"]) == 1


def test_add_source_derives_label_from_basename(tmp_path):
    folder = tmp_path / "my-journal"
    folder.mkdir()
    source = add_source(str(folder))
    assert source.label == "my-journal"


def test_add_source_disambiguates_duplicate_labels(tmp_path):
    a = tmp_path / "aaa" / "notes"
    a.mkdir(parents=True)
    b = tmp_path / "bbb" / "notes"
    b.mkdir(parents=True)

    s1 = add_source(str(a))
    s2 = add_source(str(b))

    assert s1.label == "notes"
    assert s2.label != "notes"
    assert s2.label == "bbb-notes"


def test_add_source_rejects_nonexistent_dir(tmp_path):
    with pytest.raises(ValueError, match="Not a directory"):
        add_source(str(tmp_path / "nope"))


def test_add_source_rejects_duplicate_path(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    add_source(str(folder))

    with pytest.raises(ValueError, match="Already tracked"):
        add_source(str(folder))


def test_remove_source_by_label(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    add_source(str(folder))

    removed = remove_source("notes")
    assert removed.label == "notes"
    assert load_sources() == []


def test_remove_source_nonexistent_raises():
    with pytest.raises(ValueError, match="No source with label"):
        remove_source("nope")


def test_roundtrip_load_save(tmp_path):
    sources = [
        Source(path=str(tmp_path / "a"), label="a"),
        Source(path=str(tmp_path / "b"), label="b"),
    ]
    save_sources(sources)
    loaded = load_sources()
    assert loaded == sources


def test_unique_label_fallback_numeric():
    existing = {"notes", "parent-notes"}
    label = _unique_label(Path("/x/parent/notes"), existing)
    assert label == "notes-2"
