import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", PROJECT_ROOT / ".cache" / "project-embed")).resolve()

INCLUDE_EXTENSIONS = {".py", ".ts", ".tsx", ".md"}

EXCLUDE_DIRS = {
    "node_modules",
    ".venv",
    "dist",
    "coverage",
    "build",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".claude",
    ".cursor",
    ".vscode",
    ".specstory",
    ".cache",
    "data",
}

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

CHUNK_SIZE_CHARS = 1600
CHUNK_OVERLAP_CHARS = 400

DEFAULT_TOP_K = 10

COLLECTION_NAME = "project_chunks"

RRF_K = 60
KEYWORD_WEIGHT = 3.0
