"""Text-matching scorer for image annotations."""

import re


def score_annotations(query: str, annotations: list[str]) -> float:
    """Score how well annotations match a query via word overlap.

    Returns the best match across all annotations: 0.0 (no match) to 1.0 (full match).
    Score is the fraction of query tokens found in the annotation.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0

    best = 0.0
    for annotation in annotations:
        ann_tokens = set(_tokenize(annotation))
        if not ann_tokens:
            continue
        overlap = query_tokens & ann_tokens
        if not overlap:
            continue
        recall = len(overlap) / len(query_tokens)
        best = max(best, recall)
    return best


def _tokenize(text: str) -> list[str]:
    """Lowercase split on non-alphanumeric boundaries."""
    return re.findall(r"[a-z0-9]+", text.lower())
