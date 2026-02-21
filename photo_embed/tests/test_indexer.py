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
    files, _meta = index.scan_files()
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
    assert stats == {"new": 0, "changed": 0, "deleted": 0, "total": 2}


@patch("indexer.get_device", return_value="cpu")
def test_refresh_detects_deleted(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    paths[1].unlink()
    stats = index.refresh()
    assert stats["deleted"] == 1
    assert stats["total"] == 2
    assert index._embeddings["mock-model"].shape == (2, 4)


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
def test_annotations_cleaned_on_delete(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    index.annotate(str(paths[1]), "will be deleted")
    paths[1].unlink()
    index.refresh()

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


# -- full reindex --

@patch("indexer.get_device", return_value="cpu")
def test_full_reindex(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 4)
    index.add_folder(str(photos_dir))

    stats = index.full_reindex()
    assert stats["images"] == 4
    assert "mock-model" in stats["models"]
    assert stats["elapsed"] > 0


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
def test_meta_embeddings_shrink_on_delete(_dev, tmp_path, monkeypatch):
    index = _make_index(tmp_path, monkeypatch)

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 3)
    index.add_folder(str(photos_dir))
    index.refresh()

    paths[1].unlink()
    index.refresh()
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
