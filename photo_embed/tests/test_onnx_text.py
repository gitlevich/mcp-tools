"""Tests for ONNX text encoder export and inference."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onnx_text import OnnxTextEncoder, load_text_encoder


# -- OnnxTextEncoder --


def test_loaded_false_before_load():
    enc = OnnxTextEncoder(Path("/fake.onnx"), "ViT-B-16")
    assert not enc.loaded


def test_loaded_true_after_session_set():
    enc = OnnxTextEncoder(Path("/fake.onnx"), "ViT-B-16")
    enc._session = MagicMock()
    assert enc.loaded


def test_encode_text_normalizes():
    enc = OnnxTextEncoder(Path("/fake.onnx"), "ViT-B-16")

    # Mock session returning unnormalized vector [3, 4] (norm=5)
    raw_output = np.array([[3.0, 4.0]], dtype=np.float32)
    session = MagicMock()
    session.run.return_value = [raw_output]
    enc._session = session

    # Mock tokenizer
    tok = MagicMock()
    tok.return_value = MagicMock(
        numpy=MagicMock(return_value=np.array([[1, 2, 3]], dtype=np.int64))
    )
    enc._tokenizer = tok

    vec = enc.encode_text("test")
    assert vec.shape == (2,)
    assert vec.dtype == np.float32
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
    assert abs(vec[0] - 0.6) < 1e-5  # 3/5
    assert abs(vec[1] - 0.8) < 1e-5  # 4/5


def test_encode_text_zero_vector():
    """Zero vector should stay zero (not divide by zero)."""
    enc = OnnxTextEncoder(Path("/fake.onnx"), "ViT-B-16")

    session = MagicMock()
    session.run.return_value = [np.zeros((1, 4), dtype=np.float32)]
    enc._session = session

    tok = MagicMock()
    tok.return_value = MagicMock(
        numpy=MagicMock(return_value=np.array([[1]], dtype=np.int64))
    )
    enc._tokenizer = tok

    vec = enc.encode_text("test")
    assert np.allclose(vec, 0.0)


# -- load_text_encoder --


def test_load_missing_files(tmp_path):
    assert load_text_encoder("clip-vit-b-16", tmp_path) is None


def test_load_missing_onnx_only(tmp_path):
    meta = tmp_path / "clip-vit-b-16-text.json"
    meta.write_text(json.dumps({"tokenizer_model": "ViT-B-16"}))
    assert load_text_encoder("clip-vit-b-16", tmp_path) is None


def test_load_missing_meta_only(tmp_path):
    (tmp_path / "clip-vit-b-16-text.onnx").write_bytes(b"fake")
    assert load_text_encoder("clip-vit-b-16", tmp_path) is None


def test_load_with_valid_files(tmp_path):
    (tmp_path / "clip-vit-b-16-text.onnx").write_bytes(b"fake")
    (tmp_path / "clip-vit-b-16-text.json").write_text(
        json.dumps({"tokenizer_model": "ViT-B-16"})
    )

    enc = load_text_encoder("clip-vit-b-16", tmp_path)
    assert enc is not None
    assert not enc.loaded
    assert enc._tokenizer_model == "ViT-B-16"


def test_load_corrupt_meta(tmp_path):
    (tmp_path / "clip-vit-b-16-text.onnx").write_bytes(b"fake")
    (tmp_path / "clip-vit-b-16-text.json").write_text("not json{{{")
    assert load_text_encoder("clip-vit-b-16", tmp_path) is None
