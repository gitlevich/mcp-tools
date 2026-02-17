import re
from dataclasses import dataclass
from pathlib import Path

from config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


@dataclass
class Chunk:
    text: str
    file_path: str
    rel_path: str
    start_line: int
    end_line: int
    file_type: str
    chunk_index: int


def _file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in (".ts", ".tsx"):
        return "typescript"
    if ext == ".md":
        return "markdown"
    return "unknown"


def _line_count(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def _overlap_split(text: str, start_line: int) -> list[tuple[str, int, int]]:
    """Split text exceeding CHUNK_SIZE_CHARS into overlapping pieces.
    Returns list of (chunk_text, start_line, end_line).
    """
    pieces = []
    offset = 0
    current_line = start_line

    while offset < len(text):
        end = min(offset + CHUNK_SIZE_CHARS, len(text))
        fragment = text[offset:end]

        frag_lines = _line_count(fragment)
        pieces.append((fragment, current_line, current_line + frag_lines - 1))

        if end >= len(text):
            break

        step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
        stepped_text = text[offset : offset + step]
        current_line += _line_count(stepped_text) - 1
        offset += step

    return pieces


def _chunk_code(text: str) -> list[tuple[str, int, int]]:
    """Split code on double-newline boundaries, merge small segments,
    enforce size limit with overlap.
    Returns list of (chunk_text, start_line, end_line).
    """
    segments = re.split(r"\n\n", text)
    if not segments:
        return []

    merged: list[tuple[str, int, int]] = []
    current_text = ""
    current_start = 1
    line_cursor = 1

    for seg in segments:
        sep = "\n\n" if current_text else ""
        candidate = current_text + sep + seg

        if len(candidate) <= CHUNK_SIZE_CHARS:
            current_text = candidate
        else:
            if current_text:
                end_line = current_start + _line_count(current_text) - 1
                merged.append((current_text, current_start, end_line))
            current_start = line_cursor
            current_text = seg

        line_cursor += _line_count(seg) + 2  # +2 for the \n\n separator

    if current_text:
        end_line = current_start + _line_count(current_text) - 1
        merged.append((current_text, current_start, end_line))

    # Enforce size limit on any segment that's still too large
    result = []
    for chunk_text, start, _end in merged:
        if len(chunk_text) <= CHUNK_SIZE_CHARS:
            result.append((chunk_text, start, _end))
        else:
            result.extend(_overlap_split(chunk_text, start))

    return result


_HEADING_RE = re.compile(r"^(#{1,6})\s", re.MULTILINE)


def _chunk_markdown(text: str) -> list[tuple[str, int, int]]:
    """Split markdown on heading boundaries.
    Returns list of (chunk_text, start_line, end_line).
    """
    splits = list(_HEADING_RE.finditer(text))

    if not splits:
        # No headings — treat as a single chunk or overlap-split if large
        if len(text) <= CHUNK_SIZE_CHARS:
            return [(text, 1, _line_count(text))]
        return _overlap_split(text, 1)

    sections: list[tuple[str, int]] = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        section_text = text[start:end].rstrip("\n")
        start_line = text[:start].count("\n") + 1
        sections.append((section_text, start_line))

    # Include any preamble before the first heading
    if splits[0].start() > 0:
        preamble = text[: splits[0].start()].rstrip("\n")
        if preamble.strip():
            sections.insert(0, (preamble, 1))

    result = []
    for section_text, start_line in sections:
        if len(section_text) <= CHUNK_SIZE_CHARS:
            end_line = start_line + _line_count(section_text) - 1
            result.append((section_text, start_line, end_line))
        else:
            result.extend(_overlap_split(section_text, start_line))

    return result


_SPECSTORY_DIR = ".specstory"

# Patterns for stripping specstory noise
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_USE_RE = re.compile(r"<tool-use[^>]*>.*?</tool-use>", re.DOTALL)
_DETAILS_RE = re.compile(r"<details>.*?</details>", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HR_RE = re.compile(r"^---\s*$", re.MULTILINE)
_BLANK_RUNS_RE = re.compile(r"\n{3,}")

_HTML_ENTITIES = {
    "&#96;": "`",
    "&#39;": "'",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
}


def _is_specstory(path: Path) -> bool:
    """Check if a file lives under .specstory/."""
    return _SPECSTORY_DIR in path.parts


def strip_specstory(text: str) -> str:
    """Strip specstory markdown to conversational text only."""
    text = _HTML_COMMENT_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    text = _TOOL_USE_RE.sub("", text)
    text = _DETAILS_RE.sub("", text)
    text = _HR_RE.sub("", text)
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    text = _BLANK_RUNS_RE.sub("\n\n", text)
    return text.strip()


def chunk_file(path: Path, project_root: Path) -> list[Chunk]:
    """Read a file and split it into chunks with metadata."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not text.strip():
        return []

    file_type = _file_type(path)
    abs_path = str(path.resolve())
    rel_path = str(path.resolve().relative_to(project_root.resolve()))

    if _is_specstory(path):
        text = strip_specstory(text)
        if not text.strip():
            return []

    if file_type == "markdown":
        raw_chunks = _chunk_markdown(text)
    else:
        raw_chunks = _chunk_code(text)

    return [
        Chunk(
            text=chunk_text,
            file_path=abs_path,
            rel_path=rel_path,
            start_line=start,
            end_line=end,
            file_type=file_type,
            chunk_index=i,
        )
        for i, (chunk_text, start, end) in enumerate(raw_chunks)
    ]
