import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ner import extract_entities


def test_extracts_person():
    entities = extract_entities("I met Eric Schmidt at the conference last week.")
    names = {e.name for e in entities if e.type == "person"}
    assert "Eric Schmidt" in names


def test_extracts_org():
    entities = extract_entities("She works at Google and previously was at Microsoft.")
    names = {e.name for e in entities if e.type == "org"}
    assert "Google" in names
    assert "Microsoft" in names


def test_extracts_place():
    entities = extract_entities("The event was held in San Francisco, California.")
    names = {e.name for e in entities if e.type == "place"}
    assert any("San Francisco" in n or "California" in n for n in names)


def test_filters_single_char():
    entities = extract_entities("I spoke to A about the project.")
    names = {e.name for e in entities}
    assert "A" not in names


def test_deduplicates_within_text():
    entities = extract_entities(
        "Eric went to the store. Then Eric came back. Eric was tired."
    )
    eric_entities = [e for e in entities if "Eric" in e.name]
    assert len(eric_entities) == 1


def test_returns_char_offsets():
    text = "Barack Obama was the president."
    entities = extract_entities(text)
    person = next((e for e in entities if e.type == "person"), None)
    assert person is not None
    assert text[person.char_start : person.char_end] == person.name


def test_empty_text():
    assert extract_entities("") == []


def test_no_entities():
    entities = extract_entities("The quick brown fox jumps over the lazy dog.")
    assert len(entities) == 0
