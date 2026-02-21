"""Convert ChatGPT conversations.json export into individual markdown files."""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _sanitize_filename(title: str) -> str:
    """Turn a conversation title into a safe filename."""
    clean = re.sub(r'[^\w\s-]', '', title).strip()
    clean = re.sub(r'\s+', '-', clean)
    return clean[:80] if clean else "untitled"


def _extract_messages(conversation: dict) -> list[tuple[str, str]]:
    """Walk the conversation tree from current_node to root, return (role, text) pairs."""
    mapping = conversation.get("mapping", {})
    current = conversation.get("current_node")
    chain = []

    while current and current in mapping:
        node = mapping[current]
        msg = node.get("message")
        if msg:
            role = msg.get("author", {}).get("role", "unknown")
            parts = msg.get("content", {}).get("parts", [])
            text = "".join(p for p in parts if isinstance(p, str)).strip()
            if text:
                chain.append((role, text))
        current = node.get("parent")

    chain.reverse()
    return chain


def _format_conversation(title: str, create_time: float | None, messages: list[tuple[str, str]]) -> str:
    """Format a conversation as markdown."""
    lines = [f"# {title}", ""]

    if create_time:
        dt = datetime.fromtimestamp(create_time, tz=timezone.utc)
        lines.append(f"*{dt.strftime('%Y-%m-%d')}*")
        lines.append("")

    for role, text in messages:
        if role == "system":
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"## {label}")
        lines.append("")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def convert(input_path: Path, output_dir: Path) -> int:
    """Convert conversations.json to individual .md files.

    Returns the number of conversations written.
    """
    with open(input_path) as f:
        conversations = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: dict[str, int] = {}
    written = 0

    for convo in conversations:
        title = convo.get("title") or "Untitled"
        messages = _extract_messages(convo)

        if not messages:
            continue

        base = _sanitize_filename(title)
        count = used_names.get(base, 0)
        used_names[base] = count + 1
        filename = f"{base}-{count}.md" if count > 0 else f"{base}.md"

        content = _format_conversation(title, convo.get("create_time"), messages)
        (output_dir / filename).write_text(content)
        written += 1

    return written


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <conversations.json> <output_dir>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    count = convert(input_path, output_dir)
    logger.info("Wrote %d conversations to %s", count, output_dir)
