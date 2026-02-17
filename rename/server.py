"""MCP server exposing a single rename_symbol tool."""

import logging

from mcp.server.fastmcp import FastMCP

from config import PROJECT_ROOT, TSCONFIG_PATH, detect_language
from rename_python import RenameResult as PyResult
from rename_python import rename_python
from rename_ts import RenameResult as TsResult
from rename_ts import rename_ts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("rename")


@mcp.tool()
def rename_symbol(file_path: str, old_name: str, new_name: str) -> dict:
    """Rename a symbol (function, class, variable, interface) across the codebase.

    AST-aware: Python uses Rope, TypeScript uses ts-morph.
    Applies changes directly to disk — one call, all references updated.

    Args:
        file_path: Absolute path to the file containing the symbol definition.
        old_name: Current name of the symbol.
        new_name: Desired new name.

    Returns:
        dict with old_name, new_name, and files_changed (list of paths).
    """
    lang = detect_language(file_path)
    logger.info(
        "Renaming %s -> %s in %s (%s)", old_name, new_name, file_path, lang
    )

    if lang == "python":
        result: PyResult | TsResult = rename_python(
            file_path, old_name, new_name, str(PROJECT_ROOT)
        )
    elif lang == "typescript":
        result = rename_ts(
            file_path, old_name, new_name, str(TSCONFIG_PATH)
        )
    else:
        raise ValueError(f"Unsupported language: {lang}")

    logger.info("Renamed across %d files: %s", len(result.files_changed), result.files_changed)

    return {
        "old_name": result.old_name,
        "new_name": result.new_name,
        "files_changed": result.files_changed,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
