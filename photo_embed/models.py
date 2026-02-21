import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EmbeddingModel(ABC):
    """Base interface for all embedding models."""

    name: str
    embedding_dim: int
    supports_text: bool = True

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load model weights onto device."""

    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode images to normalized embedding vectors. Returns (N, D) float32."""

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query to normalized embedding vector. Returns (D,) float32."""
        raise NotImplementedError(f"{self.name} does not support text encoding")

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts. Returns (N, D) float32."""
        return np.vstack([self.encode_text(t) for t in texts])

    @property
    def loaded(self) -> bool:
        return False


class OpenCLIPModel(EmbeddingModel):
    """CLIP and SigLIP models via the open_clip library."""

    supports_text = True

    def __init__(self, name: str, model_name: str, pretrained: str, embedding_dim: int):
        self.name = name
        self.embedding_dim = embedding_dim
        self._model_name = model_name
        self._pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, device: torch.device) -> None:
        import open_clip

        self._device = device
        logger.info("Loading %s (%s) on %s", self.name, self._model_name, device)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained or None
        )
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        self._model = self._model.to(device)
        self._model.eval()
        logger.info("Loaded %s", self.name)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self._preprocess(img) for img in images]).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensors)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)


class DINOv2Model(EmbeddingModel):
    """DINOv2 vision-only model via HuggingFace transformers."""

    supports_text = False

    def __init__(self, name: str, model_name: str, embedding_dim: int):
        self.name = name
        self.embedding_dim = embedding_dim
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._device = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self._device = device
        logger.info("Loading %s (%s) on %s", self.name, self._model_name, device)
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model = self._model.to(device)
        self._model.eval()
        logger.info("Loaded %s", self.name)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            cls_tokens = torch.nn.functional.normalize(cls_tokens, dim=-1)
        return cls_tokens.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: dict[str, EmbeddingModel] = {}


def _register_defaults() -> None:
    defaults = [
        OpenCLIPModel("clip-vit-b-16", "ViT-B-16", "openai", 512),
        OpenCLIPModel("clip-vit-l-14", "ViT-L-14", "openai", 768),
        OpenCLIPModel("siglip-vit-b-16", "ViT-B-16-SigLIP", "webli", 768),
        OpenCLIPModel(
            "siglip2-so400m-384",
            "hf-hub:timm/ViT-SO400M-16-SigLIP2-384",
            "",
            1152,
        ),
        DINOv2Model("dinov2-base", "facebook/dinov2-base", 768),
        DINOv2Model("dinov2-large", "facebook/dinov2-large", 1024),
    ]
    for m in defaults:
        AVAILABLE_MODELS[m.name] = m


_register_defaults()
