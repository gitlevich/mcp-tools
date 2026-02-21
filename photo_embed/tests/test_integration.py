"""Integration tests with real CLIP model. Downloads weights on first run."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import OpenCLIPModel, get_device


@pytest.fixture(scope="module")
def clip_model():
    model = OpenCLIPModel("clip-vit-b-16", "ViT-B-16", "openai", 512)
    model.load(get_device())
    return model


def _make_color_image(color: tuple, size: tuple = (224, 224)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def test_encode_image_shape(clip_model):
    img = _make_color_image((255, 0, 0))
    emb = clip_model.encode_images([img])
    assert emb.shape == (1, 512)
    assert emb.dtype == np.float32


def test_encode_image_normalized(clip_model):
    img = _make_color_image((0, 255, 0))
    emb = clip_model.encode_images([img])
    norm = np.linalg.norm(emb[0])
    assert abs(norm - 1.0) < 1e-4


def test_encode_text_shape(clip_model):
    vec = clip_model.encode_text("a photo of a cat")
    assert vec.shape == (512,)
    assert vec.dtype == np.float32


def test_encode_text_normalized(clip_model):
    vec = clip_model.encode_text("a landscape photograph")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4


def test_batch_encode_images(clip_model):
    images = [_make_color_image(c) for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
    emb = clip_model.encode_images(images)
    assert emb.shape == (3, 512)


def test_text_image_similarity_meaningful(clip_model):
    """Red image should match 'red' text better than 'blue' text."""
    red_img = _make_color_image((255, 0, 0))
    emb = clip_model.encode_images([red_img])[0]

    red_text = clip_model.encode_text("a solid red image")
    blue_text = clip_model.encode_text("a solid blue image")

    sim_red = float(emb @ red_text)
    sim_blue = float(emb @ blue_text)

    assert sim_red > sim_blue


def test_end_to_end_search(tmp_path, monkeypatch):
    """Full pipeline: add folder, index, search with real model."""
    import config
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "cache" / "config.json")
    monkeypatch.setattr(config, "METADATA_FILE", tmp_path / "cache" / "metadata.json")
    monkeypatch.setattr(config, "THUMBNAIL_MODEL_DIR", tmp_path / "cache" / "thumbs" / "model")
    monkeypatch.setattr(config, "THUMBNAIL_PREVIEW_DIR", tmp_path / "cache" / "thumbs" / "preview")
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", tmp_path / "cache" / "embeddings")
    monkeypatch.setattr(config, "DEFAULT_MODELS", ["clip-vit-b-16"])

    import thumbnails
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_DIR", tmp_path / "cache" / "thumbs" / "model")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_MODEL_SIZE", 384)
    monkeypatch.setattr(thumbnails, "THUMBNAIL_PREVIEW_DIR", tmp_path / "cache" / "thumbs" / "preview")
    monkeypatch.setattr(thumbnails, "THUMBNAIL_PREVIEW_SIZE", 800)

    import indexer
    monkeypatch.setattr(indexer, "CACHE_DIR", tmp_path / "cache")

    from indexer import PhotoIndex

    # Create test images with distinct colors
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    colors = {
        "red.jpg": (255, 0, 0),
        "green.jpg": (0, 255, 0),
        "blue.jpg": (0, 0, 255),
        "yellow.jpg": (255, 255, 0),
        "white.jpg": (255, 255, 255),
    }
    for name, color in colors.items():
        img = Image.new("RGB", (400, 400), color=color)
        img.save(photos_dir / name)

    index = PhotoIndex(tmp_path / "cache")
    index._state.enabled_models = ["clip-vit-b-16"]
    index.add_folder(str(photos_dir))
    index.refresh()

    assert len(index._state.images) == 5
    assert index._embeddings["clip-vit-b-16"].shape == (5, 512)

    results = index.search("a red colored image", top_k=3)
    assert len(results) == 3
    # The top result should be the red image
    assert "red.jpg" in results[0].path
