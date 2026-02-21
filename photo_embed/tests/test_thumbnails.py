import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from thumbnails import open_for_embedding, thumb_key


def _make_test_image(tmp_path: Path, name: str = "photo.jpg", size: tuple = (200, 100)) -> Path:
    img = Image.new("RGB", size, color="red")
    path = tmp_path / name
    img.save(path)
    return path


def test_thumb_key_deterministic(tmp_path):
    p = tmp_path / "a.jpg"
    k1 = thumb_key(p, 1000.0)
    k2 = thumb_key(p, 1000.0)
    assert k1 == k2
    assert len(k1) == 16


def test_thumb_key_changes_with_mtime(tmp_path):
    p = tmp_path / "a.jpg"
    k1 = thumb_key(p, 1000.0)
    k2 = thumb_key(p, 2000.0)
    assert k1 != k2


def test_model_thumbnail_generates_webp(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "THUMBNAIL_MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(config, "THUMBNAIL_MODEL_SIZE", 384)
    # Re-import to pick up patched config
    import thumbnails
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_SIZE", 384)

    src = _make_test_image(tmp_path, "photo.jpg", (1000, 800))
    dest = thumbnails.model_thumbnail(src, "testkey1")
    assert dest.exists()
    assert dest.suffix == ".webp"
    thumb = Image.open(dest)
    assert max(thumb.size) <= 384


def test_preview_thumbnail_generates_webp(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "THUMBNAIL_PREVIEW_DIR", tmp_path / "preview")
    monkeypatch.setattr(config, "THUMBNAIL_PREVIEW_SIZE", 800)
    import thumbnails
    monkeypatch.setattr(thumbnails, "THUMBNAIL_PREVIEW_DIR", tmp_path / "preview")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_PREVIEW_SIZE", 800)

    src = _make_test_image(tmp_path, "photo.jpg", (2000, 1600))
    dest = thumbnails.preview_thumbnail(src, "testkey2")
    assert dest.exists()
    thumb = Image.open(dest)
    assert max(thumb.size) <= 800


def test_model_thumbnail_cached(tmp_path, monkeypatch):
    import thumbnails
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_SIZE", 384)

    src = _make_test_image(tmp_path, "photo.jpg")
    dest1 = thumbnails.model_thumbnail(src, "cachetest")
    mtime1 = dest1.stat().st_mtime
    dest2 = thumbnails.model_thumbnail(src, "cachetest")
    mtime2 = dest2.stat().st_mtime
    assert mtime1 == mtime2  # not regenerated


def test_open_for_embedding_returns_rgb(tmp_path):
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    path = tmp_path / "rgba.webp"
    img.save(path, "WEBP")
    result = open_for_embedding(path)
    assert result.mode == "RGB"


def test_thumbnail_exif_rotation(tmp_path, monkeypatch):
    """Thumbnail should apply EXIF rotation so the image is landscape after transpose."""
    import thumbnails
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_SIZE", 384)

    # Create a portrait image (100w x 200h) with EXIF orientation=6 (90 CW rotation)
    # After exif_transpose, it should become 200w x 100h
    from PIL.ExifTags import Base as ExifBase
    img = Image.new("RGB", (100, 200), color="blue")
    exif = img.getexif()
    exif[ExifBase.Orientation] = 6  # rotate 90 CW
    path = tmp_path / "rotated.jpg"
    img.save(path, exif=exif.tobytes())

    dest = thumbnails.model_thumbnail(path, "rottest")
    thumb = Image.open(dest)
    # After rotation, width should be > height
    assert thumb.width > thumb.height
