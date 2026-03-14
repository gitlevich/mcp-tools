import os
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(unquote(os.environ.get("PROJECT_ROOT", str(Path.cwd())))).resolve()
TSCONFIG_PATH = Path(unquote(os.environ.get("TSCONFIG_PATH", str(PROJECT_ROOT / "frontend" / "tsconfig.json"))))

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
