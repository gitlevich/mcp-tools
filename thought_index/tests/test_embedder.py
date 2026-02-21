import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from embedder import GpuEmbeddingFunction, get_device


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_prefers_mps_on_mac():
    device = get_device()
    if torch.backends.mps.is_available():
        assert device.type == "mps"


def test_embedding_function_returns_correct_shape():
    fn = GpuEmbeddingFunction()
    result = fn(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 384  # all-MiniLM-L6-v2 dimension


def test_embedding_function_batch():
    fn = GpuEmbeddingFunction()
    texts = ["first sentence", "second sentence", "third sentence"]
    result = fn(texts)
    assert len(result) == 3
    assert all(len(r) == 384 for r in result)


def test_embedding_function_device():
    fn = GpuEmbeddingFunction()
    if torch.backends.mps.is_available():
        assert fn.device.type == "mps"
