"""ONNX text encoder: lightweight inference for search queries.

During indexing the full PyTorch model is loaded for image embedding.
As a side effect, the text encoder is exported to ONNX format.
At search time, only the ONNX text encoder is loaded — much faster
cold start (~1-2s vs ~15s per model) and lower memory.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class OnnxTextEncoder:
    """Runs text encoding via ONNX Runtime."""

    def __init__(self, onnx_path: Path, tokenizer_model: str):
        self.onnx_path = onnx_path
        self._tokenizer_model = tokenizer_model
        self._session = None
        self._tokenizer = None

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        import onnxruntime as ort
        import open_clip

        logger.info("Loading ONNX text encoder: %s", self.onnx_path.stem)
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = open_clip.get_tokenizer(self._tokenizer_model)
        logger.info("Loaded ONNX text encoder: %s", self.onnx_path.stem)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query to normalized embedding vector. Returns (D,) float32."""
        tokens = self._tokenizer([text]).numpy().astype(np.int64)
        outputs = self._session.run(None, {"input_ids": tokens})
        vec = outputs[0][0].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


def export_text_encoder(model, onnx_dir: Path) -> Path | None:
    """Export an OpenCLIPModel's text encoder to ONNX.

    Called when the full model is loaded (during indexing). Saves the text
    encoder for fast loading at search time.

    Returns the ONNX file path, or None on failure.
    """
    import torch

    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{model.name}-text.onnx"
    meta_path = onnx_dir / f"{model.name}-text.json"

    # meta_path is the commit marker — if it exists, export is complete
    if meta_path.exists() and onnx_path.exists():
        return onnx_path

    logger.info("Exporting ONNX text encoder: %s", model.name)

    try:
        class _Wrapper(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.m = clip_model

            def forward(self, input_ids):
                features = self.m.encode_text(input_ids)
                return features / features.norm(dim=-1, keepdim=True)

        # Move to CPU for ONNX export (MPS not supported by torch.onnx)
        original_device = model._device
        model._model.cpu()

        wrapper = _Wrapper(model._model).eval()
        dummy = model._tokenizer(["a photo"]).cpu()

        # Export directly to final path (torch may create .onnx.data alongside)
        torch.onnx.export(
            wrapper, (dummy,), str(onnx_path),
            input_names=["input_ids"],
            output_names=["text_features"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "text_features": {0: "batch"},
            },
            opset_version=17,
        )

        # Write metadata last as completion marker
        meta_path.write_text(json.dumps({"tokenizer_model": model._model_name}))

        # Move back to original device
        model._model.to(original_device)

        total_size = sum(
            f.stat().st_size for f in onnx_dir.iterdir()
            if f.name.startswith(model.name)
        )
        logger.info("Exported ONNX text encoder: %s (%.1f MB)", model.name, total_size / 1e6)
        return onnx_path

    except Exception:
        logger.warning("ONNX export failed for %s", model.name, exc_info=True)
        try:
            model._model.to(model._device)
        except Exception:
            pass
        # Clean up all files for this model (including .data files)
        for p in onnx_dir.glob(f"{model.name}-text*"):
            p.unlink(missing_ok=True)
        return None


def load_text_encoder(name: str, onnx_dir: Path) -> OnnxTextEncoder | None:
    """Load an ONNX text encoder if the exported files exist."""
    onnx_path = onnx_dir / f"{name}-text.onnx"
    meta_path = onnx_dir / f"{name}-text.json"

    if not onnx_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
        return OnnxTextEncoder(onnx_path, meta["tokenizer_model"])
    except Exception:
        logger.warning("Could not read ONNX metadata for %s", name)
        return None
