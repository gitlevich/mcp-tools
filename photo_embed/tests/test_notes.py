"""Tests for note scanning, chunking, and embedding integration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notes import NoteChunk, NoteSearchResult, chunk_file, scan_notes


@pytest.fixture
def tmp_folder(tmp_path):
    """Create a folder with sample text files."""
    (tmp_path / "conversation.md").write_text(
        "# Topic One\n\nFirst paragraph about the topic.\n\n"
        "# Topic Two\n\nSecond paragraph about another topic.\n"
    )
    (tmp_path / "notes.txt").write_text(
        "Some plain text notes.\n\nAnother paragraph.\n"
    )
    (tmp_path / "image.jpg").write_bytes(b"\xff\xd8\xff")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config")
    return tmp_path


class TestScanNotes:
    def test_finds_md_and_txt(self, tmp_folder):
        found = scan_notes([str(tmp_folder)])
        names = {p.name for p in found}
        assert "conversation.md" in names
        assert "notes.txt" in names

    def test_excludes_images(self, tmp_folder):
        found = scan_notes([str(tmp_folder)])
        names = {p.name for p in found}
        assert "image.jpg" not in names

    def test_excludes_git(self, tmp_folder):
        found = scan_notes([str(tmp_folder)])
        paths = [str(p) for p in found]
        assert not any(".git" in p for p in paths)

    def test_empty_folder_list(self):
        assert scan_notes([]) == []

    def test_nonexistent_folder(self):
        assert scan_notes(["/nonexistent/path"]) == []


class TestChunkFile:
    def test_markdown_heading_split(self, tmp_folder):
        path = tmp_folder / "conversation.md"
        chunks = chunk_file(path, str(tmp_folder))
        assert len(chunks) == 2
        assert all(isinstance(c, NoteChunk) for c in chunks)
        assert "Topic One" in chunks[0].text
        assert "Topic Two" in chunks[1].text

    def test_plaintext_single_chunk(self, tmp_folder):
        path = tmp_folder / "notes.txt"
        chunks = chunk_file(path, str(tmp_folder))
        assert len(chunks) == 1
        assert "plain text" in chunks[0].text

    def test_chunk_metadata(self, tmp_folder):
        path = tmp_folder / "conversation.md"
        chunks = chunk_file(path, str(tmp_folder))
        chunk = chunks[0]
        assert chunk.path == str(path.resolve())
        assert chunk.folder == str(tmp_folder)
        assert chunk.chunk_index == 0
        assert chunk.start_line >= 1
        assert chunk.mtime > 0
        assert tmp_folder.name in chunk.rel_path

    def test_empty_file_returns_no_chunks(self, tmp_path):
        path = tmp_path / "empty.md"
        path.write_text("")
        assert chunk_file(path, str(tmp_path)) == []

    def test_large_file_overlap_split(self, tmp_path):
        # Create a file larger than NOTE_CHUNK_SIZE
        path = tmp_path / "large.txt"
        path.write_text("word " * 2000)  # ~10000 chars
        chunks = chunk_file(path, str(tmp_path))
        assert len(chunks) > 1

    def test_rel_path_format(self, tmp_folder):
        path = tmp_folder / "conversation.md"
        chunks = chunk_file(path, str(tmp_folder))
        # rel_path should be "folder_name/filename.md"
        assert chunks[0].rel_path.endswith("conversation.md")
