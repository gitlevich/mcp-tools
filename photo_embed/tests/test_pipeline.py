import sys
from pathlib import Path
from unittest.mock import MagicMock

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

    def encode_texts(texts):
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    model.encode_images = encode_images
    model.encode_texts = encode_texts
    model.load = MagicMock()
    return model


def _patch_config(monkeypatch, tmp_path):
    import config
    import thumbnails

    cache = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "THUMBNAIL_MODEL_DIR", cache / "thumbs" / "model")
    monkeypatch.setattr(config, "THUMBNAIL_PREVIEW_DIR", cache / "thumbs" / "preview")
    monkeypatch.setattr(config, "SAVE_INTERVAL", 3)
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "BATCH_PAUSE_SECONDS", 0)

    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_DIR", cache / "thumbs" / "model")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_SIZE", 64)
    monkeypatch.setattr(thumbnails, "THUMBNAIL_PREVIEW_DIR", cache / "thumbs" / "preview")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_PREVIEW_SIZE", 64)


def test_pipeline_produces_checkpoints(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 5)

    mock = _mock_model()
    checkpoints = list(run_pipeline(
        paths=paths,
        folder_lookup=lambda p: str(photos_dir),
        models={"mock-model": mock},
    ))

    assert len(checkpoints) >= 1
    total_records = sum(len(cp.records) for cp in checkpoints)
    assert total_records == 5

    for cp in checkpoints:
        assert "mock-model" in cp.embeddings
        assert cp.embeddings["mock-model"].shape == (len(cp.records), 4)


def test_pipeline_empty_input(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    checkpoints = list(run_pipeline(
        paths=[],
        folder_lookup=lambda p: "x",
        models={"mock-model": _mock_model()},
    ))
    assert checkpoints == []


def test_pipeline_multiple_models(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 3)

    models = {
        "model-a": _mock_model("model-a", dim=4),
        "model-b": _mock_model("model-b", dim=8),
    }
    checkpoints = list(run_pipeline(
        paths=paths,
        folder_lookup=lambda p: str(photos_dir),
        models=models,
    ))

    total_records = sum(len(cp.records) for cp in checkpoints)
    assert total_records == 3

    for cp in checkpoints:
        assert "model-a" in cp.embeddings
        assert "model-b" in cp.embeddings
        n = len(cp.records)
        assert cp.embeddings["model-a"].shape == (n, 4)
        assert cp.embeddings["model-b"].shape == (n, 8)


def test_pipeline_reports_progress(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 6)
    paths = list(photos_dir.glob("*.jpg"))

    progress_calls = []

    def on_progress(current, total, detail):
        progress_calls.append((current, total, detail))

    list(run_pipeline(
        paths=paths,
        folder_lookup=lambda p: str(photos_dir),
        models={"mock-model": _mock_model()},
        progress_callback=on_progress,
    ))

    assert len(progress_calls) >= 2  # at least "Starting" + one checkpoint
    assert progress_calls[0] == (0, 6, "Starting")

    # Last progress call should have current == total
    last = progress_calls[-1]
    assert last[0] == last[1]


def test_pipeline_checkpoint_rate(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 4)
    paths = list(photos_dir.glob("*.jpg"))

    checkpoints = list(run_pipeline(
        paths=paths,
        folder_lookup=lambda p: str(photos_dir),
        models={"mock-model": _mock_model()},
    ))

    for cp in checkpoints:
        assert cp.rate >= 0
        assert cp.total_target == 4
        assert cp.total_done <= 4


def test_pipeline_handles_bad_files(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    good = _make_photos(photos_dir, 2)

    # Create a corrupt file
    bad = photos_dir / "corrupt.jpg"
    bad.write_bytes(b"not an image")

    all_paths = good + [bad]
    checkpoints = list(run_pipeline(
        paths=all_paths,
        folder_lookup=lambda p: str(photos_dir),
        models={"mock-model": _mock_model()},
    ))

    total = sum(len(cp.records) for cp in checkpoints)
    assert total == 2  # only the good ones


def test_pipeline_records_have_correct_fields(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 1)
    paths = list(photos_dir.glob("*.jpg"))

    checkpoints = list(run_pipeline(
        paths=paths,
        folder_lookup=lambda p: "test-folder",
        models={"mock-model": _mock_model()},
    ))

    record = checkpoints[0].records[0]
    assert record.folder == "test-folder"
    assert record.path.endswith(".jpg")
    assert record.mtime > 0
    assert record.size > 0
    assert len(record.thumb_key) == 16
    assert isinstance(record.metadata_text, str)


def test_pipeline_meta_embeddings(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 3)
    paths = list(photos_dir.glob("*.jpg"))

    checkpoints = list(run_pipeline(
        paths=paths,
        folder_lookup=lambda p: str(photos_dir),
        models={"mock-model": _mock_model()},
    ))

    for cp in checkpoints:
        n = len(cp.records)
        assert "mock-model" in cp.meta_embeddings
        assert cp.meta_embeddings["mock-model"].shape == (n, 4)


def test_pipeline_meta_embeddings_vision_only_model(tmp_path, monkeypatch):
    """Vision-only models should not produce meta_embeddings."""
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_photos(photos_dir, 2)
    paths = list(photos_dir.glob("*.jpg"))

    model = _mock_model("vision-only", dim=8)
    model.supports_text = False

    checkpoints = list(run_pipeline(
        paths=paths,
        folder_lookup=lambda p: str(photos_dir),
        models={"vision-only": model},
    ))

    for cp in checkpoints:
        assert "vision-only" not in cp.meta_embeddings


def test_pipeline_metadata_lookup(tmp_path, monkeypatch):
    """Extra metadata from Apple Photos should flow into records."""
    _patch_config(monkeypatch, tmp_path)

    from pipeline import run_pipeline

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = _make_photos(photos_dir, 2)

    lookup = {
        str(paths[0].resolve()): {
            "title": "Birthday Party",
            "people": ["Alice"],
        }
    }

    checkpoints = list(run_pipeline(
        paths=paths,
        folder_lookup=lambda p: str(photos_dir),
        models={"mock-model": _mock_model()},
        metadata_lookup=lookup,
    ))

    all_records = [r for cp in checkpoints for r in cp.records]
    target = str(paths[0].resolve())
    target_record = next(r for r in all_records if r.path == target)
    assert "Birthday Party" in target_record.metadata_text
    assert "Alice" in target_record.metadata_text
