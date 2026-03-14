import os
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(unquote(os.environ.get("PROJECT_ROOT", str(Path.cwd())))).resolve()
STORE_DIR = Path(unquote(os.environ.get("STORE_DIR", str(PROJECT_ROOT / ".cache" / "project-embed")))).resolve()
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local").strip().lower()

INCLUDE_EXTENSIONS = {
    ".swift",
    ".py",
    ".ts",
    ".tsx",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".pbxproj",
}

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

EMBEDDING_MODEL = "all-MiniLM-L6-v2" if EMBEDDING_PROVIDER == "local" else "text-embedding-3-small"
EMBEDDING_DIM = 384 if EMBEDDING_PROVIDER == "local" else 1536

CHUNK_SIZE_CHARS = 8000
CHUNK_OVERLAP_CHARS = 400

DEFAULT_TOP_K = 10

RRF_K = 60
KEYWORD_WEIGHT = 3.0
