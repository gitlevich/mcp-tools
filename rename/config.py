import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
TSCONFIG_PATH = Path(os.environ.get("TSCONFIG_PATH", PROJECT_ROOT / "frontend" / "tsconfig.json"))

PYTHON_EXTENSIONS = {".py"}
TYPESCRIPT_EXTENSIONS = {".ts", ".tsx"}
SWIFT_EXTENSIONS = {".swift"}


def detect_language(file_path: str) -> str:
    """Return 'python', 'typescript', or 'swift'. Raises ValueError for unsupported."""
    suffix = Path(file_path).suffix.lower()
    if suffix in PYTHON_EXTENSIONS:
        return "python"
    if suffix in TYPESCRIPT_EXTENSIONS:
        return "typescript"
    if suffix in SWIFT_EXTENSIONS:
        return "swift"
    raise ValueError(f"Unsupported file extension: {suffix}")
