"""Note chunking and scanning for embedding in CLIP/SigLIP space."""

import re
from dataclasses import dataclass
from pathlib import Path

from config import NOTE_CHUNK_OVERLAP, NOTE_CHUNK_SIZE, NOTE_EXCLUDE_DIRS, NOTE_EXTENSIONS


@dataclass
class NoteChunk:
    path: str
    folder: str
    mtime: float
    rel_path: str
    chunk_index: int
    start_line: int
    end_line: int
    text: str


@dataclass
class NoteSearchResult:
    text: str
    rel_path: str
    start_line: int
    end_line: int
    score: float


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def scan_notes(folders: list[str]) -> list[Path]:
    """Walk folders for text files, excluding common non-content dirs."""
    found = []
    for folder in folders:
        root = Path(folder)
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if any(part in NOTE_EXCLUDE_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in NOTE_EXTENSIONS and path.is_file():
                found.append(path)
    return found


# ---------------------------------------------------------------------------
# Chunking (adapted from thought_index/chunker.py)
# ---------------------------------------------------------------------------


def _line_count(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def _overlap_split(text: str, start_line: int) -> list[tuple[str, int, int]]:
    pieces = []
    offset = 0
    current_line = start_line

    while offset < len(text):
        end = min(offset + NOTE_CHUNK_SIZE, len(text))
        fragment = text[offset:end]
        frag_lines = _line_count(fragment)
        pieces.append((fragment, current_line, current_line + frag_lines - 1))

        if end >= len(text):
            break

        step = NOTE_CHUNK_SIZE - NOTE_CHUNK_OVERLAP
        stepped_text = text[offset : offset + step]
        current_line += _line_count(stepped_text) - 1
        offset += step

    return pieces


def _chunk_plaintext(text: str) -> list[tuple[str, int, int]]:
    """Split on blank-line boundaries, merge small paragraphs."""
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

        if len(candidate) <= NOTE_CHUNK_SIZE:
            current_text = candidate
        else:
            if current_text:
                end_line = current_start + _line_count(current_text) - 1
                merged.append((current_text, current_start, end_line))
            current_start = line_cursor
            current_text = seg

        line_cursor += _line_count(seg) + 2

    if current_text:
        end_line = current_start + _line_count(current_text) - 1
        merged.append((current_text, current_start, end_line))

    result = []
    for chunk_text, start, _end in merged:
        if len(chunk_text) <= NOTE_CHUNK_SIZE:
            result.append((chunk_text, start, _end))
        else:
            result.extend(_overlap_split(chunk_text, start))

    return result


_HEADING_RE = re.compile(r"^(#{1,6})\s", re.MULTILINE)


def _chunk_markdown(text: str) -> list[tuple[str, int, int]]:
    splits = list(_HEADING_RE.finditer(text))

    if not splits:
        if len(text) <= NOTE_CHUNK_SIZE:
            return [(text, 1, _line_count(text))]
        return _overlap_split(text, 1)

    sections: list[tuple[str, int]] = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        section_text = text[start:end].rstrip("\n")
        start_line = text[:start].count("\n") + 1
        sections.append((section_text, start_line))

    if splits[0].start() > 0:
        preamble = text[: splits[0].start()].rstrip("\n")
        if preamble.strip():
            sections.insert(0, (preamble, 1))

    result = []
    for section_text, start_line in sections:
        if len(section_text) <= NOTE_CHUNK_SIZE:
            end_line = start_line + _line_count(section_text) - 1
            result.append((section_text, start_line, end_line))
        else:
            result.extend(_overlap_split(section_text, start_line))

    return result


def chunk_file(path: Path, folder: str) -> list[NoteChunk]:
    """Read a text file and split into NoteChunks."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not text.strip():
        return []

    is_markdown = path.suffix.lower() == ".md"
    raw_chunks = _chunk_markdown(text) if is_markdown else _chunk_plaintext(text)

    folder_path = Path(folder)
    try:
        inner = str(path.resolve().relative_to(folder_path.resolve()))
    except ValueError:
        inner = path.name
    folder_name = folder_path.name
    rel_path = f"{folder_name}/{inner}"
    mtime = path.stat().st_mtime

    return [
        NoteChunk(
            path=str(path.resolve()),
            folder=folder,
            mtime=mtime,
            rel_path=rel_path,
            chunk_index=i,
            start_line=start,
            end_line=end,
            text=chunk_text,
        )
        for i, (chunk_text, start, end) in enumerate(raw_chunks)
    ]
