import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotations import score_annotations, _tokenize


def test_tokenize_basic():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_punctuation():
    assert _tokenize("that's my father!") == ["that", "s", "my", "father"]


def test_tokenize_numbers():
    assert _tokenize("summer 2019") == ["summer", "2019"]


def test_exact_match():
    score = score_annotations("my father", ["my father on the beach"])
    assert score == 1.0


def test_partial_match():
    score = score_annotations("father beach sunset", ["my father on the beach"])
    # "father" and "beach" match (2/3)
    assert abs(score - 2 / 3) < 0.01


def test_no_match():
    score = score_annotations("mountain hiking", ["beach sunset"])
    assert score == 0.0


def test_empty_annotations():
    score = score_annotations("anything", [])
    assert score == 0.0


def test_empty_query():
    score = score_annotations("", ["some annotation"])
    assert score == 0.0


def test_case_insensitive():
    score = score_annotations("Father", ["my father at the park"])
    assert score == 1.0


def test_multiple_annotations_best_wins():
    score = score_annotations(
        "red car",
        ["beach sunset", "a red car in the parking lot"],
    )
    assert score == 1.0


def test_single_word_query():
    score = score_annotations("father", ["that is my father"])
    assert score == 1.0


def test_no_overlap_different_words():
    score = score_annotations("cat dog", ["bird fish"])
    assert score == 0.0
