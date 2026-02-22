"""Named entity extraction using spaCy."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "PERSON": "person",
    "ORG": "org",
    "GPE": "place",
    "LOC": "place",
}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp

    import spacy

    try:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        logger.info("Downloading spaCy model en_core_web_sm ...")
        from spacy.cli import download

        download("en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

    return _nlp


@dataclass
class ExtractedEntity:
    name: str
    type: str
    char_start: int
    char_end: int


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract named entities from text.

    Returns deduplicated entities with their character offsets.
    Maps spaCy labels to: person, place, org.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    results: list[ExtractedEntity] = []
    seen: set[tuple[str, str]] = set()

    for ent in doc.ents:
        mapped_type = _LABEL_MAP.get(ent.label_)
        if mapped_type is None:
            continue

        name = ent.text.strip()
        if len(name) < 2:
            continue

        key = (name.lower(), mapped_type)
        if key in seen:
            continue
        seen.add(key)

        results.append(
            ExtractedEntity(
                name=name,
                type=mapped_type,
                char_start=ent.start_char,
                char_end=ent.end_char,
            )
        )

    return results
