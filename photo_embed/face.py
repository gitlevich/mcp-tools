"""Face detection and recognition using insightface (ArcFace).

Provides FaceRecord dataclass for face metadata and FaceEngine for
detection + embedding. The engine uses buffalo_l (RetinaFace + ArcFace)
via ONNX Runtime on CPU.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceRecord:
    image_path: str
    face_idx: int
    bbox: tuple[float, float, float, float]  # normalized (x1, y1, x2, y2), 0-1
    confidence: float
    label: str = ""


class FaceEngine:
    """Detect faces and compute ArcFace embeddings using insightface."""

    def __init__(self):
        self._app = None

    def load(self) -> None:
        import insightface

        self._app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("FaceEngine loaded (buffalo_l, CPU)")

    @property
    def loaded(self) -> bool:
        return self._app is not None

    def detect_and_embed(
        self, image_path: str
    ) -> list[tuple[FaceRecord, np.ndarray]]:
        """Detect all faces in an image and return (FaceRecord, embedding) pairs.

        Args:
            image_path: Path to image file (JPEG, PNG, HEIF, or WebP thumbnail).

        Returns:
            List of (FaceRecord, 512-dim float32 embedding) tuples.
        """
        img = self._load_bgr(image_path)
        if img is None:
            return []

        faces = self._app.get(img)
        h, w = img.shape[:2]
        results = []
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            normalized_bbox = (
                round(float(max(0.0, x1 / w)), 6),
                round(float(max(0.0, y1 / h)), 6),
                round(float(min(1.0, x2 / w)), 6),
                round(float(min(1.0, y2 / h)), 6),
            )
            record = FaceRecord(
                image_path=str(Path(image_path).resolve()),
                face_idx=i,
                bbox=normalized_bbox,
                confidence=round(float(face.det_score), 4),
            )
            embedding = face.normed_embedding.astype(np.float32)
            results.append((record, embedding))

        return results

    @staticmethod
    def _load_bgr(path: str) -> np.ndarray | None:
        """Load an image as BGR numpy array (insightface convention)."""
        try:
            import cv2

            img = cv2.imread(path)
            if img is not None:
                return img
        except Exception:
            pass

        # Fallback for formats cv2 can't read (HEIF, WebP)
        try:
            from PIL import Image

            pil_img = Image.open(path).convert("RGB")
            arr = np.array(pil_img)
            return arr[:, :, ::-1].copy()  # RGB -> BGR
        except Exception:
            logger.debug("Failed to load image: %s", path)
            return None
