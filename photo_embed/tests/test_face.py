import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face import FaceEngine, FaceRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_photos(folder: Path, count: int = 5) -> list[Path]:
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
    import config
    import thumbnails

    cache = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "CONFIG_FILE", cache / "config.json")
    monkeypatch.setattr(config, "METADATA_FILE", cache / "metadata.json")
    monkeypatch.setattr(config, "ANNOTATIONS_FILE", cache / "annotations.json")
    monkeypatch.setattr(config, "FACES_FILE", cache / "faces.json")
    monkeypatch.setattr(config, "FACES_SCANNED_FILE", cache / "faces_scanned.txt")
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

    monkeypatch.setattr(indexer, "DEFAULT_MODELS", list(model_names))

    models = {}
    for name in model_names:
        m = _mock_model(name)
        models[name] = m
    monkeypatch.setattr(indexer, "AVAILABLE_MODELS", models)
    monkeypatch.setattr(indexer, "CACHE_DIR", cache)


def _build_index_with_faces(tmp_path, monkeypatch, n_images=5, faces_per_image=2):
    """Build a PhotoIndex with synthetic face data injected."""
    _patch_all(monkeypatch, tmp_path)

    import config
    from indexer import PhotoIndex

    photos = tmp_path / "photos"
    photos.mkdir()
    paths = _make_photos(photos, n_images)

    with patch("indexer.get_device", return_value="cpu"):
        idx = PhotoIndex(config.CACHE_DIR)
        idx.add_folder(str(photos))
        idx.refresh()

    # Inject synthetic face records and embeddings
    rng = np.random.RandomState(42)
    face_records = []
    face_embeddings = []

    for img_i, img_rec in enumerate(idx._state.images):
        for face_j in range(faces_per_image):
            rec = FaceRecord(
                image_path=img_rec.path,
                face_idx=face_j,
                bbox=(0.2, 0.2, 0.5, 0.5),
                confidence=0.99,
                label="",
            )
            face_records.append(rec)
            emb = rng.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            face_embeddings.append(emb)

    # Make faces 0 and 2 (same person in images 0 and 1) very similar
    if len(face_embeddings) >= 3:
        face_embeddings[2] = face_embeddings[0] + rng.randn(512).astype(np.float32) * 0.05
        face_embeddings[2] /= np.linalg.norm(face_embeddings[2])

    idx._face_records = face_records
    idx._face_embeddings = np.vstack(face_embeddings).astype(np.float32)
    idx._rebuild_face_lookups()
    idx._save_faces()

    return idx, paths


# ---------------------------------------------------------------------------
# Tests: FaceRecord
# ---------------------------------------------------------------------------


def test_face_record_dataclass():
    rec = FaceRecord(
        image_path="/tmp/test.jpg",
        face_idx=0,
        bbox=(0.1, 0.2, 0.3, 0.4),
        confidence=0.95,
        label="Alice",
    )
    d = asdict(rec)
    assert d["image_path"] == "/tmp/test.jpg"
    assert d["label"] == "Alice"
    assert len(d["bbox"]) == 4


# ---------------------------------------------------------------------------
# Tests: FaceEngine
# ---------------------------------------------------------------------------


def test_face_engine_load_bgr_jpeg(tmp_path):
    """Test that _load_bgr can read a JPEG file."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    p = tmp_path / "test.jpg"
    img.save(p)

    arr = FaceEngine._load_bgr(str(p))
    assert arr is not None
    assert arr.shape == (200, 200, 3)
    # BGR format: check blue channel (last in RGB, first in BGR)
    # Allow JPEG compression rounding
    assert abs(int(arr[100, 100, 0]) - 200) <= 2


def test_face_engine_load_bgr_missing_file():
    arr = FaceEngine._load_bgr("/nonexistent/path.jpg")
    assert arr is None


# ---------------------------------------------------------------------------
# Tests: PhotoIndex face methods
# ---------------------------------------------------------------------------


@patch("indexer.get_device", return_value="cpu")
def test_find_person_returns_results(_dev, tmp_path, monkeypatch):
    """find_person should return images with similar faces."""
    idx, paths = _build_index_with_faces(tmp_path, monkeypatch)

    # Find person using face 0 of image 0 (similar to face 0 of image 1)
    results = idx.find_person(str(paths[0]), face_idx=0)
    assert len(results) > 0
    # The most similar face should be from image 1 (faces 0 and 2 are similar)
    assert results[0].path == str(paths[1].resolve())
    assert results[0].score > 0.5


@patch("indexer.get_device", return_value="cpu")
def test_find_person_unknown_image(_dev, tmp_path, monkeypatch):
    """find_person with unknown path should return empty."""
    idx, _ = _build_index_with_faces(tmp_path, monkeypatch)
    results = idx.find_person("/nonexistent/photo.jpg")
    assert results == []


@patch("indexer.get_device", return_value="cpu")
def test_find_person_no_faces(_dev, tmp_path, monkeypatch):
    """find_person on an index with no face data should return empty."""
    _patch_all(monkeypatch, tmp_path)

    import config
    from indexer import PhotoIndex

    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, 3)

    idx = PhotoIndex(config.CACHE_DIR)
    idx.add_folder(str(photos))
    idx.refresh()

    results = idx.find_person(str(list(photos.iterdir())[0]))
    assert results == []


@patch("indexer.get_device", return_value="cpu")
def test_label_face(_dev, tmp_path, monkeypatch):
    """label_face should update face record and add @person annotation."""
    idx, paths = _build_index_with_faces(tmp_path, monkeypatch)

    msg = idx.label_face(str(paths[0]), face_idx=0, label="Alice")
    assert "Alice" in msg

    # Verify face record updated
    faces = idx.get_faces(str(paths[0]))
    labeled = [f for f in faces if f["label"] == "Alice"]
    assert len(labeled) == 1
    assert labeled[0]["face_idx"] == 0

    # Verify @person annotation added
    annotations = idx.get_annotations(str(paths[0]))
    assert "@person Alice" in annotations

    # Verify people list
    people = idx.get_people()
    names = [p["name"] for p in people]
    assert "Alice" in names


@patch("indexer.get_device", return_value="cpu")
def test_label_face_not_found(_dev, tmp_path, monkeypatch):
    """label_face with wrong face_idx should return error message."""
    idx, paths = _build_index_with_faces(tmp_path, monkeypatch)

    msg = idx.label_face(str(paths[0]), face_idx=99, label="Bob")
    assert "not found" in msg.lower()


@patch("indexer.get_device", return_value="cpu")
def test_batch_label_faces(_dev, tmp_path, monkeypatch):
    """batch_label_faces should label multiple faces at once."""
    idx, paths = _build_index_with_faces(tmp_path, monkeypatch)

    matches = [
        {"image_path": str(paths[0]), "face_idx": 0},
        {"image_path": str(paths[1]), "face_idx": 0},
    ]
    stats = idx.batch_label_faces(matches, "Dasha")
    assert stats["labeled"] == 2
    assert stats["label"] == "Dasha"

    # Verify both faces labeled
    for p in [paths[0], paths[1]]:
        faces = idx.get_faces(str(p))
        labeled = [f for f in faces if f["label"] == "Dasha"]
        assert len(labeled) == 1

    # Verify people count
    people = idx.get_people()
    dasha = [p for p in people if p["name"] == "Dasha"][0]
    assert dasha["image_count"] == 2


@patch("indexer.get_device", return_value="cpu")
def test_get_faces(_dev, tmp_path, monkeypatch):
    """get_faces should return all face records for an image."""
    idx, paths = _build_index_with_faces(tmp_path, monkeypatch, faces_per_image=3)

    faces = idx.get_faces(str(paths[0]))
    assert len(faces) == 3
    assert all(f["image_path"] == str(paths[0].resolve()) for f in faces)
    assert [f["face_idx"] for f in faces] == [0, 1, 2]


@patch("indexer.get_device", return_value="cpu")
def test_face_persistence(_dev, tmp_path, monkeypatch):
    """Face data should survive save/load cycle."""
    idx, paths = _build_index_with_faces(tmp_path, monkeypatch)
    idx.label_face(str(paths[0]), face_idx=0, label="TestPerson")

    # Reload index from disk
    import config
    from indexer import PhotoIndex

    idx2 = PhotoIndex(config.CACHE_DIR)

    # Verify face data loaded
    assert len(idx2._face_records) == len(idx._face_records)
    assert idx2._face_embeddings is not None
    assert idx2._face_embeddings.shape == idx._face_embeddings.shape

    # Verify label persisted
    faces = idx2.get_faces(str(paths[0]))
    labeled = [f for f in faces if f["label"] == "TestPerson"]
    assert len(labeled) == 1

    # Verify find_person still works
    results = idx2.find_person(str(paths[0]), face_idx=0)
    assert len(results) > 0


@patch("indexer.get_device", return_value="cpu")
def test_get_people_empty(_dev, tmp_path, monkeypatch):
    """get_people should return empty list when no faces are labeled."""
    idx, _ = _build_index_with_faces(tmp_path, monkeypatch)
    assert idx.get_people() == []


@patch("indexer.get_device", return_value="cpu")
def test_status_includes_face_data(_dev, tmp_path, monkeypatch):
    """status() should include face counts."""
    idx, paths = _build_index_with_faces(tmp_path, monkeypatch)
    idx.label_face(str(paths[0]), face_idx=0, label="Alice")

    s = idx.status()
    assert s["total_faces"] == 10  # 5 images * 2 faces
    assert s["labeled_faces"] == 1
    assert s["people"] == 1


# ---------------------------------------------------------------------------
# Tests: refresh_faces streaming pipeline
# ---------------------------------------------------------------------------


def _mock_face_engine(faces_per_image=2, dim=512):
    """Return a mock FaceEngine that produces deterministic faces."""
    rng = np.random.RandomState(99)
    engine = MagicMock()
    engine.loaded = True

    call_count = [0]

    def detect_and_embed(source):
        results = []
        for j in range(faces_per_image):
            rec = FaceRecord(
                image_path=source,
                face_idx=j,
                bbox=(0.1, 0.1, 0.5, 0.5),
                confidence=0.95,
            )
            emb = rng.randn(dim).astype(np.float32)
            emb /= np.linalg.norm(emb)
            results.append((rec, emb))
        call_count[0] += 1
        return results

    engine.detect_and_embed = detect_and_embed
    engine._call_count = call_count
    return engine


def _setup_face_test(tmp_path, monkeypatch, n_images=5, faces_per_image=2):
    """Build a PhotoIndex ready for refresh_faces with pre-filter bypassed."""
    _patch_all(monkeypatch, tmp_path)

    import config
    from indexer import PhotoIndex

    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, n_images)

    idx = PhotoIndex(config.CACHE_DIR)
    idx.add_folder(str(photos))
    idx.refresh()

    engine = _mock_face_engine(faces_per_image=faces_per_image)
    idx._face_engine = engine
    idx._ensure_face_engine = lambda: engine
    # Bypass CLIP pre-filter so all candidates pass through
    idx._prefilter_people = lambda candidates, **kw: candidates

    return idx, photos, config, engine


@patch("indexer.get_device", return_value="cpu")
def test_refresh_faces_streaming(_dev, tmp_path, monkeypatch):
    """refresh_faces should detect faces and populate records/embeddings."""
    idx, photos, config, engine = _setup_face_test(
        tmp_path, monkeypatch, n_images=5, faces_per_image=2
    )

    stats = idx.refresh_faces()

    assert stats["images_scanned"] == 5
    assert stats["new_faces"] == 10  # 5 images * 2 faces
    assert stats["total_faces"] == 10
    assert len(idx._face_records) == 10
    assert idx._face_embeddings is not None
    assert idx._face_embeddings.shape == (10, 512)
    assert len(idx._face_scanned_paths) == 5


@patch("indexer.get_device", return_value="cpu")
def test_refresh_faces_progress_callback(_dev, tmp_path, monkeypatch):
    """Progress callback should be called with (current, total, detail)."""
    idx, photos, config, engine = _setup_face_test(
        tmp_path, monkeypatch, n_images=3, faces_per_image=1
    )

    calls = []
    idx.refresh_faces(progress_callback=lambda c, t, d: calls.append((c, t, d)))

    assert len(calls) == 3
    # All calls should have 3 elements and correct total
    for c, t, d in calls:
        assert t == 3
        assert isinstance(d, str)
    # current should be 1, 2, 3
    assert [c for c, _, _ in calls] == [1, 2, 3]


@patch("indexer.get_device", return_value="cpu")
def test_refresh_faces_idempotent(_dev, tmp_path, monkeypatch):
    """Second refresh_faces call should find nothing new."""
    idx, photos, config, engine = _setup_face_test(
        tmp_path, monkeypatch, n_images=3, faces_per_image=1
    )

    stats1 = idx.refresh_faces()
    assert stats1["new_faces"] == 3

    stats2 = idx.refresh_faces()
    assert stats2["new_faces"] == 0
    assert stats2["images_scanned"] == 0


@patch("indexer.get_device", return_value="cpu")
def test_refresh_faces_persistence(_dev, tmp_path, monkeypatch):
    """Face data should survive save/reload with new scanned paths format."""
    idx, photos, config, engine = _setup_face_test(
        tmp_path, monkeypatch, n_images=4, faces_per_image=2
    )
    idx.refresh_faces()

    # Verify faces_scanned.txt exists on disk
    scanned_file = config.CACHE_DIR / "faces_scanned.txt"
    assert scanned_file.exists()
    lines = [l.strip() for l in scanned_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 4

    # Reload and verify
    from indexer import PhotoIndex

    idx2 = PhotoIndex(config.CACHE_DIR)
    assert len(idx2._face_records) == 8
    assert idx2._face_embeddings is not None
    assert idx2._face_embeddings.shape == (8, 512)
    assert len(idx2._face_scanned_paths) == 4

    # Second refresh should find nothing (pre-filter bypassed on idx2 too)
    idx2._face_engine = engine
    idx2._ensure_face_engine = lambda: engine
    idx2._prefilter_people = lambda candidates, **kw: candidates
    stats = idx2.refresh_faces()
    assert stats["new_faces"] == 0


@patch("indexer.get_device", return_value="cpu")
def test_refresh_faces_buffer_growth(_dev, tmp_path, monkeypatch):
    """Buffer should grow when more faces found than estimated."""
    idx, photos, config, engine = _setup_face_test(
        tmp_path, monkeypatch, n_images=2, faces_per_image=50
    )

    stats = idx.refresh_faces()
    assert stats["new_faces"] == 100
    assert idx._face_embeddings.shape == (100, 512)


@patch("indexer.get_device", return_value="cpu")
def test_load_faces_backward_compat(_dev, tmp_path, monkeypatch):
    """Old-format faces.json with 'scanned' key should load correctly."""
    _patch_all(monkeypatch, tmp_path)

    import config
    from indexer import PhotoIndex

    photos = tmp_path / "photos"
    photos.mkdir()
    paths = _make_photos(photos, 2)

    idx = PhotoIndex(config.CACHE_DIR)
    idx.add_folder(str(photos))
    idx.refresh()

    # Write old-format faces.json with "scanned" key
    cache = config.CACHE_DIR
    old_data = {
        "records": [
            {
                "image_path": str(paths[0].resolve()),
                "face_idx": 0,
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "confidence": 0.95,
                "label": "",
            }
        ],
        "scanned": [str(p.resolve()) for p in paths],
    }
    (cache / "faces.json").write_text(json.dumps(old_data))

    rng = np.random.RandomState(42)
    emb = rng.randn(1, 512).astype(np.float32)
    (cache / "embeddings").mkdir(parents=True, exist_ok=True)
    np.save(str(cache / "embeddings" / "faces.npy"), emb)

    # Reload -- should pick up scanned from old JSON
    idx2 = PhotoIndex(cache)
    assert len(idx2._face_records) == 1
    assert len(idx2._face_scanned_paths) == 2
    assert idx2._face_embeddings is not None

    # refresh_faces should find nothing new (all scanned)
    engine = _mock_face_engine()
    idx2._face_engine = engine
    idx2._ensure_face_engine = lambda: engine
    stats = idx2.refresh_faces()
    assert stats["new_faces"] == 0


@patch("indexer.get_device", return_value="cpu")
def test_refresh_faces_detection_failure(_dev, tmp_path, monkeypatch):
    """Failed detection on one image should not stop scanning."""
    idx, photos, config, _ = _setup_face_test(
        tmp_path, monkeypatch, n_images=3, faces_per_image=1
    )

    call_count = [0]

    def flaky_detect(source):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("simulated crash")
        rec = FaceRecord(
            image_path=source,
            face_idx=0,
            bbox=(0.1, 0.1, 0.5, 0.5),
            confidence=0.9,
        )
        emb = np.random.randn(512).astype(np.float32)
        return [(rec, emb)]

    engine = MagicMock()
    engine.loaded = True
    engine.detect_and_embed = flaky_detect
    idx._face_engine = engine
    idx._ensure_face_engine = lambda: engine

    stats = idx.refresh_faces()
    # Image 2 failed, but images 1 and 3 should succeed
    assert stats["new_faces"] == 2
    assert stats["images_scanned"] == 3
    # All 3 paths should be marked scanned (even the failed one)
    assert len(idx._face_scanned_paths) == 3
