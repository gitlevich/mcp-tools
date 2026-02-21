import os
from pathlib import Path

CACHE_DIR = Path(
    os.environ.get("PHOTO_CACHE_DIR", Path.home() / ".cache" / "photo-embed")
).resolve()

THUMBNAILS_DIR = CACHE_DIR / "thumbnails"
THUMBNAIL_MODEL_DIR = THUMBNAILS_DIR / "model"
THUMBNAIL_PREVIEW_DIR = THUMBNAILS_DIR / "preview"
EMBEDDINGS_DIR = CACHE_DIR / "embeddings"
CONFIG_FILE = CACHE_DIR / "config.json"
METADATA_FILE = CACHE_DIR / "metadata.json"
ANNOTATIONS_FILE = CACHE_DIR / "annotations.json"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heif", ".heic"}

THUMBNAIL_MODEL_SIZE = 384
THUMBNAIL_PREVIEW_SIZE = 800

EMBEDDING_BATCH_SIZE = 16
BATCH_PAUSE_SECONDS = 0.1
SAVE_INTERVAL = 100  # save state every N images during indexing
NICE_VALUE = 15

DEFAULT_TOP_K = 10

DEFAULT_MODELS = ["clip-vit-b-16", "siglip-vit-b-16"]

# -- service daemon --

SERVICE_PORT = int(os.environ.get("PHOTO_EMBED_PORT", "7820"))
SERVICE_HOST = "127.0.0.1"
SERVICE_PID_FILE = Path("/tmp/mcp-tools/photo-embed.pid")
SERVICE_STARTUP_TIMEOUT = 60  # seconds to wait for health check
