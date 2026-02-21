import hashlib
import logging
from pathlib import Path

from PIL import Image, ImageOps

# Allow large panoramas — we thumbnail immediately so memory is freed
Image.MAX_IMAGE_PIXELS = None

from config import (
    THUMBNAIL_MODEL_DIR,
    THUMBNAIL_MODEL_SIZE,
    THUMBNAIL_PREVIEW_DIR,
    THUMBNAIL_PREVIEW_SIZE,
)

logger = logging.getLogger(__name__)

_heif_registered = False


def register_heif() -> None:
    """Register HEIF/HEIC opener with Pillow (idempotent)."""
    global _heif_registered
    if _heif_registered:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        _heif_registered = True
        logger.info("HEIF support registered")
    except ImportError:
        logger.warning("pillow-heif not installed; HEIF/HEIC files will be skipped")


def thumb_key(path: Path, mtime: float) -> str:
    """Deterministic cache key from path and mtime."""
    raw = f"{path}:{mtime}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _generate(source: Path, dest: Path, size: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        img.save(dest, "WEBP", quality=85)
    return dest


def model_thumbnail(source: Path, key: str) -> Path:
    """Return model-ready thumbnail (384px), generating if missing."""
    dest = THUMBNAIL_MODEL_DIR / f"{key}.webp"
    if dest.exists():
        return dest
    return _generate(source, dest, THUMBNAIL_MODEL_SIZE)


def preview_thumbnail(source: Path, key: str) -> Path:
    """Return display preview thumbnail (800px), generating if missing."""
    dest = THUMBNAIL_PREVIEW_DIR / f"{key}.webp"
    if dest.exists():
        return dest
    return _generate(source, dest, THUMBNAIL_PREVIEW_SIZE)


def open_for_embedding(thumbnail_path: Path) -> Image.Image:
    """Open a cached model thumbnail as RGB PIL Image.

    Forces pixel data into memory and closes the file handle immediately
    so the pipeline doesn't accumulate tens of thousands of open fds.
    """
    img = Image.open(thumbnail_path)
    img.load()  # read pixels into memory, release file handle
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
