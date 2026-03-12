#!/usr/bin/env python3
"""CLI wrapper for AST-aware rename across Python, TypeScript, and Swift.

Usage:
    rename_cli.py <file_path> <old_name> <new_name>

Outputs JSON: {"old_name": "...", "new_name": "...", "files_changed": [...]}
On error:     {"error": "message"}
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROJECT_ROOT, TSCONFIG_PATH, detect_language
from rename_python import rename_python
from rename_swift import rename_swift
from rename_ts import rename_ts


def main() -> None:
    if len(sys.argv) != 4:
        print(json.dumps({"error": "Usage: rename_cli.py <file_path> <old_name> <new_name>"}))
        sys.exit(1)

    file_path, old_name, new_name = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        lang = detect_language(file_path)

        if lang == "python":
            result = rename_python(file_path, old_name, new_name, str(PROJECT_ROOT))
        elif lang == "typescript":
            result = rename_ts(file_path, old_name, new_name, str(TSCONFIG_PATH))
        elif lang == "swift":
            result = rename_swift(file_path, old_name, new_name)
        else:
            raise ValueError(f"Unsupported language: {lang}")

        print(json.dumps({
            "old_name": result.old_name,
            "new_name": result.new_name,
            "files_changed": result.files_changed,
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
