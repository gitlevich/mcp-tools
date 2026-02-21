import os
from pathlib import Path

CONFIG_DIR = Path(
    os.environ.get("THOUGHT_INDEX_CONFIG_DIR", Path.home() / ".config" / "thought-index")
).resolve()

CHROMA_DIR = CONFIG_DIR / "chroma"
SOURCES_FILE = CONFIG_DIR / "sources.json"

INCLUDE_EXTENSIONS = {".md", ".txt"}

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}

CHUNK_SIZE_CHARS = 1600
CHUNK_OVERLAP_CHARS = 400

DEFAULT_TOP_K = 10
MAX_PER_FILE = 2
DIVERSITY_OVERFETCH = 5  # fetch N * top_k from Chroma before applying diversity

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "thought_chunks"

REFRESH_DEBOUNCE_SECONDS = 60
