import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_photos(folder: Path, count: int = 3) -> list[Path]:
    paths = []
    for i in range(count):
        img = Image.new("RGB", (100, 100), color=(i * 50, 100, 200))
        p = folder / f"photo_{i}.jpg"
        img.save(p)
        paths.append(p)
    return paths


def _mock_model(name: str = "mock-model", dim: int = 4):
    model = MagicMock()
    model.name = name
    model.embedding_dim = dim
    model.supports_text = True
    model.loaded = True

    def encode_images(images):
        vecs = np.random.randn(len(images), dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def encode_text(text):
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def encode_texts(texts):
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    model.encode_images = encode_images
    model.encode_text = encode_text
    model.encode_texts = encode_texts
    model.load = MagicMock()
    return model


def _patch_all(monkeypatch, tmp_path, model_names=("mock-model",)):
    """Patch config and thumbnails for isolated testing."""
    import config
    import thumbnails

    cache = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "CONFIG_FILE", cache / "config.json")
    monkeypatch.setattr(config, "METADATA_FILE", cache / "metadata.json")
    monkeypatch.setattr(config, "ANNOTATIONS_FILE", cache / "annotations.json")
    monkeypatch.setattr(config, "THUMBNAIL_MODEL_DIR", cache / "thumbs" / "model")
    monkeypatch.setattr(config, "THUMBNAIL_PREVIEW_DIR", cache / "thumbs" / "preview")
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", cache / "embeddings")
    monkeypatch.setattr(config, "DEFAULT_MODELS", list(model_names))
    monkeypatch.setattr(config, "SAVE_INTERVAL", 100)
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 16)
    monkeypatch.setattr(config, "BATCH_PAUSE_SECONDS", 0)

    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_DIR", cache / "thumbs" / "model")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_SIZE", 64)
    monkeypatch.setattr(thumbnails, "THUMBNAIL_PREVIEW_DIR", cache / "thumbs" / "preview")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_PREVIEW_SIZE", 64)

    import indexer
    models = {}
    for name in model_names:
        m = _mock_model(name)
        models[name] = m
    monkeypatch.setattr(indexer, "AVAILABLE_MODELS", models)
    monkeypatch.setattr(indexer, "CACHE_DIR", cache)

    return cache


def _make_index(tmp_path, monkeypatch, model_name="mock-model"):
    cache = _patch_all(monkeypatch, tmp_path, model_names=(model_name,))
    from indexer import PhotoIndex
    index = PhotoIndex(cache)
    index._state.enabled_models = [model_name]
    return index


# -- folder management --

@patch("indexer.get_device", return_value="cpu")
def test_add_and_scan_folder(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)
    index._state.enabled_models = []

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 3)

    index.add_folder(str(photos_dir))
    files, _meta, _scanned = index.scan_files()
    assert len(files) == 3


@patch("indexer.get_device", return_value="cpu")
def test_folder_management(_dev, tmp_path, monkeypatch):
    _patch_all(monkeypatch, tmp_path, model_names=())
    from indexer import PhotoIndex

    d1 = tmp_path / "folder1"
    d1.mkdir()
    d2 = tmp_path / "folder2"
    d2.mkdir()

    index = PhotoIndex(tmp_path / "cache")
    index._state.enabled_models = []

    assert index.list_folders() == []
    index.add_folder(str(d1))
    index.add_folder(str(d2))
    assert len(index.list_folders()) == 2

    assert "Already" in index.add_folder(str(d1))
    index.remove_folder(str(d1))
    assert len(index.list_folders()) == 1
    assert "Not configured" in index.remove_folder(str(d1))


# -- refresh --

@patch("indexer.get_device", return_value="cpu")
def test_refresh_indexes_new_images(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))

    stats = index.refresh()
    assert stats["total"] == 3
    assert stats["new"] == 3
    assert len(index._state.images) == 3
    assert index._embeddings["mock-model"].shape == (3, 4)
    assert index._meta_embeddings["mock-model"].shape == (3, 4)


@patch("indexer.get_device", return_value="cpu")
def test_refresh_no_changes_skips_pipeline(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 2)
    index.add_folder(str(photos_dir))

    index.refresh()
    stats = index.refresh()
    assert stats["new"] == 0
    assert stats["total"] == 2


@patch("indexer.get_device", return_value="cpu")
def test_refresh_never_deletes(_dev, tmp_path, monkeypatch):
    """Deleting a file from disk must NOT remove it from the index via refresh."""
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    paths[1].unlink()
    stats = index.refresh()
    # refresh is append-only: no deletion
    assert stats["total"] == 3
    assert index._embeddings["mock-model"].shape == (3, 4)


@patch("indexer.get_device", return_value="cpu")
def test_refresh_preserves_images_from_inaccessible_folder(_dev, tmp_path, monkeypatch):
    """Images from an unmounted/inaccessible folder must not be deleted."""
    index = _make_index(tmp_path, monkeypatch)

    folder_a = tmp_path / "local_photos"
    folder_a.mkdir()
    _make_photos(folder_a, 2)

    folder_b = tmp_path / "external_drive"
    folder_b.mkdir()
    _make_photos(folder_b, 3)

    index.add_folder(str(folder_a))
    index.add_folder(str(folder_b))
    index.refresh()
    assert len(index._state.images) == 5
    assert index._embeddings["mock-model"].shape == (5, 4)

    # Simulate unmounted drive by removing the directory
    import shutil
    shutil.rmtree(folder_b)

    stats = index.refresh()
    assert stats["total"] == 5
    assert index._embeddings["mock-model"].shape == (5, 4)


@patch("indexer.get_device", return_value="cpu")
def test_refresh_calls_progress_callback(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))

    calls = []
    index.refresh(progress_callback=lambda c, t, d: calls.append((c, t, d)))
    assert len(calls) >= 1
    assert calls[0] == (0, 3, "Starting")


# -- persistence --

@patch("indexer.get_device", return_value="cpu")
def test_state_persistence(_dev, tmp_path, monkeypatch):
    cache = _patch_all(monkeypatch, tmp_path)
    from indexer import PhotoIndex

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 2)

    idx1 = PhotoIndex(cache)
    idx1._state.enabled_models = ["mock-model"]
    idx1.add_folder(str(photos_dir))
    idx1.refresh()

    idx2 = PhotoIndex(cache)
    assert len(idx2._state.folders) == 1
    assert len(idx2._state.images) == 2
    assert idx2._embeddings["mock-model"].shape == (2, 4)


# -- search --

@patch("indexer.get_device", return_value="cpu")
def test_search_returns_results(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 5)
    index.add_folder(str(photos_dir))
    index.refresh()

    results = index.search("a red photo", top_k=3)
    assert len(results) == 3
    assert all(r.path for r in results)
    assert all(r.score > 0 for r in results)


@patch("indexer.get_device", return_value="cpu")
def test_search_empty_index(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)
    assert index.search("anything") == []


# -- annotations --

@patch("indexer.get_device", return_value="cpu")
def test_annotate_and_retrieve(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    assert "Annotated" in index.annotate(str(paths[0]), "my father")
    assert index.get_annotations(str(paths[0])) == ["my father"]


@patch("indexer.get_device", return_value="cpu")
def test_annotate_unknown_image(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 1)
    index.add_folder(str(photos_dir))
    index.refresh()

    result = index.annotate("/nonexistent/photo.jpg", "note")
    assert "not in index" in result.lower()


@patch("indexer.get_device", return_value="cpu")
def test_annotations_cleaned_on_prune(_dev, tmp_path, monkeypatch):
    """Annotations are cleaned when images are pruned (explicit action)."""
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    index.annotate(str(paths[1]), "will be pruned")
    paths[1].unlink()
    index.prune_deleted()

    assert index.get_annotations(str(paths[1])) == []


@patch("indexer.get_device", return_value="cpu")
def test_search_boosts_annotated(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 5)
    index.add_folder(str(photos_dir))
    index.refresh()

    target = str(paths[2].resolve())
    index.annotate(str(paths[2]), "red vintage car")

    results = index.search("red vintage car", top_k=5)
    assert results[0].path == target


# -- prune_deleted --

@patch("indexer.get_device", return_value="cpu")
def test_prune_deleted_removes_missing(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 4)
    index.add_folder(str(photos_dir))
    index.refresh()

    paths[0].unlink()
    paths[2].unlink()

    stats = index.prune_deleted()
    assert stats["pruned"] == 2
    assert stats["total"] == 2
    assert index._embeddings["mock-model"].shape == (2, 4)
    remaining_paths = {r.path for r in index._state.images}
    assert str(paths[0].resolve()) not in remaining_paths
    assert str(paths[1].resolve()) in remaining_paths


@patch("indexer.get_device", return_value="cpu")
def test_prune_deleted_noop_when_all_exist(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    stats = index.prune_deleted()
    assert stats["pruned"] == 0
    assert stats["total"] == 3


# -- full reindex --

@patch("indexer.get_device", return_value="cpu")
def test_full_reindex(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 4)
    index.add_folder(str(photos_dir))

    stats = index.full_reindex(confirm=True)
    assert stats["images"] == 4
    assert "mock-model" in stats["models"]
    assert stats["elapsed"] > 0


@patch("indexer.get_device", return_value="cpu")
def test_full_reindex_requires_confirm(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 2)
    index.add_folder(str(photos_dir))
    index.refresh()

    import pytest
    with pytest.raises(ValueError, match="confirm=True"):
        index.full_reindex()

    # Images should be untouched
    assert len(index._state.images) == 2


@patch("indexer.get_device", return_value="cpu")
def test_reindex_notes_preserves_images(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()
    assert len(index._state.images) == 3

    stats = index.reindex_notes()
    assert "notes" in stats
    # Images must remain intact
    assert len(index._state.images) == 3
    assert index._embeddings["mock-model"].shape == (3, 4)


# -- status --

@patch("indexer.get_device", return_value="cpu")
def test_status(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 2)
    index.add_folder(str(photos_dir))
    index.refresh()

    s = index.status()
    assert s["total_images"] == 2
    assert s["annotated_images"] == 0
    assert "mock-model" in s["embeddings"]


# -- meta embeddings --

@patch("indexer.get_device", return_value="cpu")
def test_meta_embeddings_persisted(_dev, tmp_path, monkeypatch):
    cache = _patch_all(monkeypatch, tmp_path)
    from indexer import PhotoIndex

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 2)

    idx1 = PhotoIndex(cache)
    idx1._state.enabled_models = ["mock-model"]
    idx1.add_folder(str(photos_dir))
    idx1.refresh()
    assert idx1._meta_embeddings["mock-model"].shape == (2, 4)

    idx2 = PhotoIndex(cache)
    assert idx2._meta_embeddings["mock-model"].shape == (2, 4)


@patch("indexer.get_device", return_value="cpu")
def test_meta_embeddings_shrink_on_prune(_dev, tmp_path, monkeypatch):
    """prune_deleted() shrinks meta embeddings to match remaining images."""
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    paths[1].unlink()
    stats = index.prune_deleted()
    assert stats["pruned"] == 1
    assert stats["total"] == 2
    assert index._meta_embeddings["mock-model"].shape == (2, 4)


@patch("indexer.get_device", return_value="cpu")
def test_meta_embeddings_in_search(_dev, tmp_path, monkeypatch):
    """Search should use meta_embeddings as an additional RRF source."""
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    results = index.search("test query", top_k=3)
    assert len(results) == 3
    # Each result should have model scores for both visual and meta
    for r in results:
        assert "mock-model" in r.model_scores
        assert "mock-model-meta" in r.model_scores


# -- progressive search --

@patch("indexer.get_device", return_value="cpu")
def test_search_progressive_yields_phases(_dev, tmp_path, monkeypatch):
    """search_progressive yields at least 2 phases with 2 models."""
    cache = _patch_all(monkeypatch, tmp_path, model_names=("model-a", "model-b"))
    from indexer import PhotoIndex

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 5)

    index = PhotoIndex(cache)
    index._state.enabled_models = ["model-a", "model-b"]
    index.add_folder(str(photos_dir))
    index.refresh()

    phases = list(index.search_progressive("sunset", top_k=3))
    assert len(phases) == 2
    # Both phases return result lists
    assert len(phases[0]) == 3
    assert len(phases[1]) == 3
    # Final phase includes scores from both models
    for r in phases[1]:
        assert "model-a" in r.model_scores
        assert "model-b" in r.model_scores


@patch("indexer.get_device", return_value="cpu")
def test_search_progressive_single_model(_dev, tmp_path, monkeypatch):
    """With 1 model, yields 2 phases (after model + after keyword signals)."""
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    phases = list(index.search_progressive("test", top_k=3))
    assert len(phases) == 2
    assert len(phases[-1]) == 3


@patch("indexer.get_device", return_value="cpu")
def test_search_progressive_empty_index(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)
    phases = list(index.search_progressive("anything"))
    assert phases == []


@patch("indexer.get_device", return_value="cpu")
def test_search_progressive_matches_search(_dev, tmp_path, monkeypatch):
    """Final phase of progressive search should match regular search."""
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 5)
    index.add_folder(str(photos_dir))
    index.refresh()

    # Fix random seed so both calls produce same results
    np.random.seed(42)
    regular = index.search("beach", top_k=5)
    np.random.seed(42)
    phases = list(index.search_progressive("beach", top_k=5))
    final = phases[-1]

    assert len(regular) == len(final)
    for r1, r2 in zip(regular, final):
        assert r1.path == r2.path
        assert r1.score == r2.score


# -- date filter parsing --

from indexer import _parse_date_filter, _year_from_date_taken


def test_parse_date_filter_single_year():
    q, y1, y2 = _parse_date_filter("New York City 2010")
    assert q == "New York City"
    assert y1 == 2010
    assert y2 == 2010


def test_parse_date_filter_year_range():
    q, y1, y2 = _parse_date_filter("beach 2015-2020")
    assert q == "beach"
    assert y1 == 2015
    assert y2 == 2020


def test_parse_date_filter_decade():
    q, y1, y2 = _parse_date_filter("wedding 2010s")
    assert q == "wedding"
    assert y1 == 2010
    assert y2 == 2019


def test_parse_date_filter_no_date():
    q, y1, y2 = _parse_date_filter("sunset at the beach")
    assert q == "sunset at the beach"
    assert y1 is None
    assert y2 is None


def test_parse_date_filter_year_at_start():
    q, y1, y2 = _parse_date_filter("2023 birthday party")
    assert q == "birthday party"
    assert y1 == 2023
    assert y2 == 2023


def test_parse_date_filter_range_reversed():
    q, y1, y2 = _parse_date_filter("photos 2020-2015")
    assert q == "photos"
    assert y1 == 2015
    assert y2 == 2020


def test_parse_date_filter_only_year():
    q, y1, y2 = _parse_date_filter("2010")
    assert q == ""
    assert y1 == 2010
    assert y2 == 2010


def test_year_from_date_taken_valid():
    assert _year_from_date_taken("2023:07:15 14:30:00") == 2023


def test_year_from_date_taken_empty():
    assert _year_from_date_taken("") is None


def test_year_from_date_taken_invalid():
    assert _year_from_date_taken("not-a-date") is None


# -- batch reverse geocode --

from indexer import _batch_reverse_geocode


def test_batch_reverse_geocode_returns_locations():
    coords = [(48.8566, 2.3522), (40.7128, -74.0060)]  # Paris, NYC
    locations = _batch_reverse_geocode(coords)
    assert len(locations) == 2
    assert "Paris" in locations[0]
    assert "New York" in locations[1]


def test_batch_reverse_geocode_empty():
    assert _batch_reverse_geocode([]) == []


# -- search results include location --

@patch("indexer.get_device", return_value="cpu")
def test_search_results_include_location(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 2)
    index.add_folder(str(photos_dir))
    index.refresh()

    # Inject GPS coordinates into image records
    index._state.images[0].latitude = 48.8566
    index._state.images[0].longitude = 2.3522

    results = index.search("test", top_k=2)
    paris_result = [r for r in results if "Paris" in r.location]
    assert len(paris_result) == 1
    no_loc_result = [r for r in results if r.location == ""]
    assert len(no_loc_result) == 1
