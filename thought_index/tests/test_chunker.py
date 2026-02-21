import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunker import Chunk, chunk_file, _chunk_markdown, _chunk_plaintext, _overlap_split
from config import CHUNK_SIZE_CHARS


def test_chunk_small_markdown():
    text = "# Title\n\nSome content here."
    chunks = _chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0][0] == "# Title\n\nSome content here."


def test_chunk_markdown_splits_on_headings():
    text = "# First\n\nContent one.\n\n# Second\n\nContent two."
    chunks = _chunk_markdown(text)
    assert len(chunks) == 2
    assert chunks[0][0].startswith("# First")
    assert chunks[1][0].startswith("# Second")


def test_chunk_markdown_preamble():
    text = "Preamble text.\n\n# Heading\n\nBody."
    chunks = _chunk_markdown(text)
    assert len(chunks) == 2
    assert chunks[0][0] == "Preamble text."
    assert chunks[1][0].startswith("# Heading")


def test_chunk_plaintext_splits_on_blank_lines():
    big = "A" * 1000
    text = f"{big}\n\n{big}\n\n{big}"
    chunks = _chunk_plaintext(text)
    assert len(chunks) >= 2


def test_chunk_plaintext_merges_small_paragraphs():
    text = "Short one.\n\nShort two.\n\nShort three."
    chunks = _chunk_plaintext(text)
    assert len(chunks) == 1
    assert "Short one." in chunks[0][0]
    assert "Short three." in chunks[0][0]


def test_chunk_file_with_source_label(tmp_path):
    f = tmp_path / "idea.md"
    f.write_text("# My idea\n\nThis is great.")

    chunks = chunk_file(f, tmp_path, "notes")
    assert len(chunks) == 1
    assert chunks[0].rel_path == "notes/idea.md"
    assert chunks[0].file_type == "markdown"


def test_chunk_file_txt_type(tmp_path):
    f = tmp_path / "thoughts.txt"
    f.write_text("Some thoughts here.")

    chunks = chunk_file(f, tmp_path, "journal")
    assert len(chunks) == 1
    assert chunks[0].file_type == "plaintext"
    assert chunks[0].rel_path == "journal/thoughts.txt"


def test_chunk_file_empty(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("")
    assert chunk_file(f, tmp_path, "x") == []


def test_chunk_file_nonexistent(tmp_path):
    f = tmp_path / "nope.md"
    assert chunk_file(f, tmp_path, "x") == []


def test_overlap_split_large_content():
    text = "A" * (CHUNK_SIZE_CHARS * 3)
    pieces = _overlap_split(text, 1)
    assert len(pieces) >= 3
    for chunk_text, start, end in pieces:
        assert len(chunk_text) <= CHUNK_SIZE_CHARS
        assert start >= 1


def test_chunk_indices_sequential(tmp_path):
    f = tmp_path / "multi.md"
    f.write_text("# One\n\nContent.\n\n# Two\n\nMore.\n\n# Three\n\nEnd.")

    chunks = chunk_file(f, tmp_path, "test")
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_line_numbers_positive(tmp_path):
    f = tmp_path / "lines.md"
    f.write_text("# Title\n\nParagraph one.\n\n# Next\n\nParagraph two.")

    chunks = chunk_file(f, tmp_path, "test")
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_chunk_file_subdirectory(tmp_path):
    sub = tmp_path / "sub" / "deep"
    sub.mkdir(parents=True)
    f = sub / "note.txt"
    f.write_text("Deep thought.")

    chunks = chunk_file(f, tmp_path, "vault")
    assert chunks[0].rel_path == "vault/sub/deep/note.txt"
