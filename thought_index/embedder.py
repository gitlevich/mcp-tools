import logging

import torch
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GpuEmbeddingFunction(EmbeddingFunction[Documents]):
    """Sentence-transformers on MPS/CUDA/CPU via Chroma's EmbeddingFunction interface."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model_name = model_name
        self._device = get_device()
        logger.info("Loading %s on %s", model_name, self._device)
        self._model = SentenceTransformer(model_name, device=str(self._device))

    def name(self) -> str:
        return f"gpu-{self._model_name}"

    @property
    def device(self) -> torch.device:
        return self._device

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self._model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()
