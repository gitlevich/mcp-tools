import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class Source:
    path: str
    label: str


def load_sources() -> list[Source]:
    if not cfg.SOURCES_FILE.exists():
        return []
    data = json.loads(cfg.SOURCES_FILE.read_text())
    return [Source(**s) for s in data.get("sources", [])]


def save_sources(sources: list[Source]) -> None:
    cfg.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {"sources": [asdict(s) for s in sources]}
    cfg.SOURCES_FILE.write_text(json.dumps(data, indent=2))


def _unique_label(path: Path, existing_labels: set[str]) -> str:
    base = path.name
    if base not in existing_labels:
        return base
    candidate = f"{path.parent.name}-{base}"
    if candidate not in existing_labels:
        return candidate
    for i in range(2, 100):
        candidate = f"{base}-{i}"
        if candidate not in existing_labels:
            return candidate
    raise ValueError(f"Cannot derive unique label for {path}")


def add_source(folder_path: str) -> Source:
    path = Path(folder_path).resolve()
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    sources = load_sources()
    existing_paths = {s.path for s in sources}
    if str(path) in existing_paths:
        raise ValueError(f"Already tracked: {path}")

    existing_labels = {s.label for s in sources}
    label = _unique_label(path, existing_labels)
    source = Source(path=str(path), label=label)
    sources.append(source)
    save_sources(sources)
    return source


def remove_source(label: str) -> Source:
    sources = load_sources()
    for i, s in enumerate(sources):
        if s.label == label:
            removed = sources.pop(i)
            save_sources(sources)
            return removed
    raise ValueError(f"No source with label: {label}")
