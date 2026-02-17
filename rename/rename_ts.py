import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve().parent / "ts_rename.js"


@dataclass
class RenameResult:
    old_name: str
    new_name: str
    files_changed: list[str]


def rename_ts(
    file_path: str,
    old_name: str,
    new_name: str,
    tsconfig_path: str,
) -> RenameResult:
    """Rename a TypeScript symbol across the project via ts-morph subprocess."""
    result = subprocess.run(
        [
            "node",
            str(SCRIPT_PATH),
            file_path,
            old_name,
            new_name,
            tsconfig_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        # Try to extract structured error from stdout
        try:
            data = json.loads(result.stdout)
            raise RuntimeError(data.get("error", "Unknown ts-morph error"))
        except json.JSONDecodeError:
            raise RuntimeError(
                f"ts_rename.js failed (exit {result.returncode}): "
                f"{result.stderr or result.stdout}"
            )

    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(data["error"])

    return RenameResult(
        old_name=old_name,
        new_name=new_name,
        files_changed=data["files_changed"],
    )
