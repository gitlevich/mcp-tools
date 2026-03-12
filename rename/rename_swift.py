"""AST-aware Swift rename via SourceKit-LSP.

SourceKit-LSP uses the build index for cross-file rename.
For Xcode projects, a recent build is needed so the index is fresh.
For SPM packages, a swift-build is triggered automatically to ensure the index exists.
"""

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SOURCEKIT_LSP = "sourcekit-lsp"


@dataclass
class RenameResult:
    old_name: str
    new_name: str
    files_changed: list[str]


def find_project_root(file_path: str) -> Path:
    """Walk up from file_path looking for Package.swift or *.xcodeproj."""
    current = Path(file_path).resolve().parent
    while current != current.parent:
        if (current / "Package.swift").exists():
            return current
        if any(current.glob("*.xcodeproj")):
            return current
        current = current.parent
    raise ValueError(
        f"No Package.swift or .xcodeproj found above {file_path}"
    )


def find_symbol_position(source: str, name: str) -> tuple[int, int]:
    """Find (line, character) of the first whole-word occurrence of name.

    Both line and character are 0-based, matching LSP convention.
    """
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Symbol '{name}' not found in source")

    offset = match.start()
    before = source[:offset]
    line = before.count("\n")
    last_newline = before.rfind("\n")
    character = offset - (last_newline + 1)
    return line, character


# ---------------------------------------------------------------------------
# Minimal JSON-RPC over stdio for LSP
# ---------------------------------------------------------------------------

class LSPClient:
    """Thin JSON-RPC client for a stdio-based LSP server."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._id = 0

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def send(self, method: str, params: dict, *, notify: bool = False) -> int | None:
        msg: dict = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notify:
            rid = self._next_id()
            msg["id"] = rid
        payload = json.dumps(msg)
        header = f"Content-Length: {len(payload)}\r\n\r\n"
        self._proc.stdin.write(header.encode())
        self._proc.stdin.write(payload.encode())
        self._proc.stdin.flush()
        return None if notify else rid

    def recv(self, expected_id: int | None = None) -> dict:
        """Read the next JSON-RPC response, skipping notifications."""
        while True:
            headers = {}
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("sourcekit-lsp closed stdout unexpectedly")
                line = line.decode("utf-8").strip()
                if line == "":
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip()] = value.strip()

            length = int(headers["Content-Length"])
            body = self._proc.stdout.read(length)
            msg = json.loads(body)

            # Skip server-initiated notifications and requests
            if "id" not in msg or (expected_id is not None and msg.get("id") != expected_id):
                if "id" in msg and "method" in msg:
                    # Server request — send empty response to unblock it
                    resp = json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": None})
                    header = f"Content-Length: {len(resp)}\r\n\r\n"
                    self._proc.stdin.write(header.encode())
                    self._proc.stdin.write(resp.encode())
                    self._proc.stdin.flush()
                if "id" not in msg:
                    continue
                if expected_id is not None and msg.get("id") != expected_id:
                    continue
            return msg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _ensure_index(project_root: Path) -> None:
    """Build SPM packages so the index exists for cross-file rename."""
    package_swift = project_root / "Package.swift"
    if not package_swift.exists():
        return  # Xcode project — user must build in Xcode
    logger.info("Building SPM package at %s to ensure index", project_root)
    subprocess.run(
        ["swift", "build"],
        cwd=str(project_root),
        capture_output=True,
        timeout=120,
    )


def rename_swift(
    file_path: str,
    old_name: str,
    new_name: str,
) -> RenameResult:
    """Rename a Swift symbol across the project via SourceKit-LSP."""
    abs_path = Path(file_path).resolve()
    project_root = find_project_root(file_path)
    _ensure_index(project_root)

    source = abs_path.read_text()
    line, character = find_symbol_position(source, old_name)

    file_uri = abs_path.as_uri()
    root_uri = project_root.as_uri()

    proc = subprocess.Popen(
        [SOURCEKIT_LSP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        client = LSPClient(proc)
        changed = _do_rename(client, root_uri, file_uri, source, line, character, new_name)
    finally:
        try:
            sid = client.send("shutdown", {})
            client.recv(sid)
            client.send("exit", {}, notify=True)
        except Exception:
            pass
        proc.terminate()
        proc.wait(timeout=5)

    return RenameResult(
        old_name=old_name,
        new_name=new_name,
        files_changed=sorted(changed),
    )


def _do_rename(
    client: LSPClient,
    root_uri: str,
    file_uri: str,
    source: str,
    line: int,
    character: int,
    new_name: str,
) -> list[str]:
    """Run the LSP initialize → rename → apply flow. Returns changed file paths."""
    # Initialize
    rid = client.send("initialize", {
        "processId": None,
        "rootUri": root_uri,
        "capabilities": {},
    })
    client.recv(rid)
    client.send("initialized", {}, notify=True)

    # Open document
    client.send("textDocument/didOpen", {
        "textDocument": {
            "uri": file_uri,
            "languageId": "swift",
            "version": 1,
            "text": source,
        },
    }, notify=True)

    # Wait for background indexer to pick up cross-file references
    time.sleep(2)

    # Rename
    rid = client.send("textDocument/rename", {
        "textDocument": {"uri": file_uri},
        "position": {"line": line, "character": character},
        "newName": new_name,
    })
    response = client.recv(rid)

    if "error" in response:
        raise RuntimeError(response["error"].get("message", str(response["error"])))

    workspace_edit = response.get("result")
    if workspace_edit is None:
        raise RuntimeError(f"Symbol '{new_name}' rename returned no edits")

    return _apply_workspace_edit(workspace_edit)


def _apply_workspace_edit(edit: dict) -> list[str]:
    """Apply a WorkspaceEdit to disk. Returns list of changed file paths."""
    changes = edit.get("changes", {})

    # Also handle documentChanges if present
    if not changes and "documentChanges" in edit:
        for doc_change in edit["documentChanges"]:
            uri = doc_change["textDocument"]["uri"]
            changes.setdefault(uri, []).extend(doc_change.get("edits", []))

    # Deduplicate URIs that resolve to the same path (macOS /var vs /private/var)
    by_path: dict[str, list] = {}
    for uri, edits in changes.items():
        resolved = _uri_to_path(uri)
        by_path.setdefault(resolved, []).extend(edits)

    changed_paths = []

    for path, edits in by_path.items():
        content = Path(path).read_text()
        lines = content.split("\n")

        # Deduplicate edits (symlink URIs can produce duplicates)
        seen = set()
        unique_edits = []
        for e in edits:
            key = (
                e["range"]["start"]["line"],
                e["range"]["start"]["character"],
                e["range"]["end"]["line"],
                e["range"]["end"]["character"],
                e["newText"],
            )
            if key not in seen:
                seen.add(key)
                unique_edits.append(e)

        # Sort edits in reverse order so earlier offsets stay valid
        sorted_edits = sorted(
            unique_edits,
            key=lambda e: (e["range"]["start"]["line"], e["range"]["start"]["character"]),
            reverse=True,
        )

        for text_edit in sorted_edits:
            start = text_edit["range"]["start"]
            end = text_edit["range"]["end"]
            new_text = text_edit["newText"]

            # Convert line/character to string offset
            start_offset = _position_to_offset(lines, start["line"], start["character"])
            end_offset = _position_to_offset(lines, end["line"], end["character"])

            flat = "\n".join(lines)
            flat = flat[:start_offset] + new_text + flat[end_offset:]
            lines = flat.split("\n")

        Path(path).write_text("\n".join(lines))
        changed_paths.append(path)

    return changed_paths


def _position_to_offset(lines: list[str], line: int, character: int) -> int:
    """Convert LSP (line, character) to a flat string offset."""
    offset = sum(len(lines[i]) + 1 for i in range(line))  # +1 for \n
    offset += character
    return offset


def _uri_to_path(uri: str) -> str:
    """Convert a file:// URI to a resolved filesystem path."""
    if uri.startswith("file://"):
        from urllib.parse import unquote
        return str(Path(unquote(uri[7:])).resolve())
    return uri
