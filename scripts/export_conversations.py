#!/usr/bin/env python3
"""Export Claude Code JSONL conversations to readable markdown.

Reads session files from ~/.claude/projects/<project-slug>/ and writes
markdown transcripts to docs/genesis/. Existing transcripts are appended
to, never overwritten.

Usage:
    python scripts/export_conversations.py
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENESIS_DIR = PROJECT_ROOT / "docs" / "genesis"
STATE_FILE = GENESIS_DIR / ".export_state.json"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

# Derived from the absolute project path, matching Claude Code's convention:
# all path separators, spaces, and underscores become dashes.
PROJECT_SLUG = re.sub(r"[/ _]", "-", str(PROJECT_ROOT)).lstrip("-")


def find_session_dir() -> Path:
    """Locate the Claude Code session directory for this project."""
    session_dir = CLAUDE_PROJECTS / f"-{PROJECT_SLUG}"
    # Claude Code encodes the path with leading dash
    if not session_dir.exists():
        # Try without the leading dash
        session_dir = CLAUDE_PROJECTS / PROJECT_SLUG
    if not session_dir.exists():
        raise FileNotFoundError(
            f"No Claude Code session directory found. Tried:\n"
            f"  {CLAUDE_PROJECTS / f'-{PROJECT_SLUG}'}\n"
            f"  {CLAUDE_PROJECTS / PROJECT_SLUG}"
        )
    return session_dir


def strip_system_tags(text: str) -> str:
    """Remove system-injected tags from user messages."""
    # Remove <system-reminder>...</system-reminder> blocks
    text = re.sub(
        r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL
    )
    # Remove <ide_opened_file>...</ide_opened_file> blocks
    text = re.sub(
        r"<ide_opened_file>.*?</ide_opened_file>", "", text, flags=re.DOTALL
    )
    # Remove <ide_selection>...</ide_selection> blocks
    text = re.sub(
        r"<ide_selection>.*?</ide_selection>", "", text, flags=re.DOTALL
    )
    return text.strip()


FILE_PATH_TOOLS = {"Read", "Write", "Edit"}
PROJECT_ROOT_STR = str(PROJECT_ROOT)


def relativize(path: str) -> str | None:
    """Convert an absolute path to a project-relative path, or None if outside."""
    if path.startswith(PROJECT_ROOT_STR):
        rel = path[len(PROJECT_ROOT_STR) :].lstrip("/")
        return rel if rel else None
    return None


def extract_file_refs(content: list[dict]) -> list[str]:
    """Extract project-relative file paths from tool_use blocks."""
    refs = []
    seen = set()
    for block in content:
        if block.get("type") != "tool_use":
            continue
        if block.get("name") not in FILE_PATH_TOOLS:
            continue
        raw = block.get("input", {}).get("file_path", "")
        rel = relativize(raw)
        if rel and rel not in seen:
            refs.append(rel)
            seen.add(rel)
    return refs


def extract_turns(jsonl_path: Path) -> list[dict]:
    """Extract conversation turns from a JSONL session file.

    Returns a list of dicts with keys: uuid, role, text, timestamp, files.
    Only includes turns with actual user/assistant text content.
    Tool-use file references are collected between text turns and attached
    to the next text turn as a 'files' list.
    """
    turns = []
    pending_files: list[str] = []

    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            msg_type = record.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            message = record.get("message", {})
            content = message.get("content", [])
            uuid = record.get("uuid", "")
            timestamp = record.get("timestamp", "")

            if not isinstance(content, list):
                continue

            # Collect file refs from tool_use blocks in this message
            pending_files.extend(extract_file_refs(content))

            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])

            combined = "\n\n".join(text_parts)
            if msg_type == "user":
                combined = strip_system_tags(combined)

            if not combined.strip():
                continue

            # Deduplicate while preserving order
            seen = set()
            unique_files = []
            for f_path in pending_files:
                if f_path not in seen:
                    unique_files.append(f_path)
                    seen.add(f_path)

            turns.append(
                {
                    "uuid": uuid,
                    "role": msg_type,
                    "text": combined.strip(),
                    "timestamp": timestamp,
                    "files": unique_files,
                }
            )
            pending_files = []

    return turns


def session_metadata(jsonl_path: Path) -> dict:
    """Extract session metadata from the first few records."""
    slug = ""
    session_id = jsonl_path.stem
    first_timestamp = ""
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if not first_timestamp and record.get("timestamp"):
                first_timestamp = record["timestamp"]
            if not slug and record.get("slug"):
                slug = record["slug"]
            if first_timestamp and slug:
                break
    return {
        "session_id": session_id,
        "slug": slug,
        "first_timestamp": first_timestamp,
    }


def markdown_filename(meta: dict) -> str:
    """Generate a markdown filename from session metadata."""
    ts = meta["first_timestamp"]
    if ts:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y-%m-%d_%H%M")
    else:
        date_str = "unknown"

    slug = meta["slug"]
    if slug:
        return f"{date_str}_{slug}.md"
    return f"{date_str}_{meta['session_id'][:8]}.md"


def load_state() -> dict:
    """Load export state tracking which UUIDs have been written per session."""
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    """Persist export state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def format_turn(turn: dict) -> str:
    """Format a single conversation turn as markdown."""
    role_label = "User" if turn["role"] == "user" else "Assistant"
    ts = turn["timestamp"]
    time_str = ""
    if ts:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        time_str = dt.strftime(" (%H:%M UTC)")

    lines = [
        "",
        f"### {role_label}{time_str}",
        "",
    ]

    if turn.get("files"):
        rel_prefix = os.path.relpath(PROJECT_ROOT, GENESIS_DIR)
        links = [
            f"[{f}]({rel_prefix}/{f})" for f in turn["files"]
        ]
        lines.append(f"*Files: {', '.join(links)}*")
        lines.append("")

    lines.append(turn["text"])
    lines.append("")

    return "\n".join(lines)


def format_header(meta: dict) -> str:
    """Format the markdown file header."""
    ts = meta["first_timestamp"]
    if ts:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    else:
        date_str = "unknown"

    slug = meta["slug"] or meta["session_id"][:8]
    lines = [
        f"# Session: {slug}",
        f"**Date**: {date_str}  ",
        f"**Session ID**: `{meta['session_id']}`",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def export_session(jsonl_path: Path, state: dict) -> None:
    """Export a single JSONL session to markdown, appending new turns."""
    meta = session_metadata(jsonl_path)
    filename = markdown_filename(meta)
    md_path = GENESIS_DIR / filename

    turns = extract_turns(jsonl_path)
    if not turns:
        log.info("Skipping %s (no conversation content)", jsonl_path.name)
        return

    existing_uuids = set(state.get(filename, []))
    new_turns = [t for t in turns if t["uuid"] not in existing_uuids]
    if not new_turns:
        log.info("Up to date: %s", filename)
        return

    is_new_file = not md_path.exists() or md_path.stat().st_size == 0

    with open(md_path, "a") as f:
        if is_new_file:
            f.write(format_header(meta))

        for turn in new_turns:
            f.write(format_turn(turn))

    state[filename] = list(existing_uuids | {t["uuid"] for t in new_turns})

    log.info(
        "%s %s (%d new turn%s)",
        "Created" if is_new_file else "Appended to",
        filename,
        len(new_turns),
        "s" if len(new_turns) != 1 else "",
    )


def main():
    session_dir = find_session_dir()
    GENESIS_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(session_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.info("No session files found in %s", session_dir)
        return

    log.info("Found %d session(s) in %s", len(jsonl_files), session_dir)

    state = load_state()
    for jsonl_path in jsonl_files:
        export_session(jsonl_path, state)
    save_state(state)


if __name__ == "__main__":
    main()
