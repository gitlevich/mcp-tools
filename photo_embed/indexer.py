import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import reverse_geocode

from annotations import score_annotations
from config import (
    ANNOTATIONS_FILE,
    CACHE_DIR,
    CONFIG_FILE,
    DEFAULT_MODELS,
    ENTITIES_DB,
    FACE_SIMILARITY_THRESHOLD,
    FACES_FILE,
    FACES_SCANNED_FILE,
    METADATA_FILE,
    NOTE_MAX_PER_FILE,
    NOTES_METADATA_FILE,
    SEARCH_SCORE_RATIO,
    SUPPORTED_EXTENSIONS,
)
from entities import EntityStore
from face import FaceEngine, FaceRecord
from models import AVAILABLE_MODELS, EmbeddingModel, OpenCLIPModel, get_device
from onnx_text import OnnxTextEncoder, export_text_encoder, load_text_encoder
from notes import NoteChunk, NoteSearchResult, chunk_file, scan_notes
from pipeline import Checkpoint, ImageRecord, ProgressCallback, run_pipeline

logger = logging.getLogger(__name__)


def _batch_reverse_geocode(coords: list[tuple[float, float]]) -> list[str]:
    """Reverse geocode a batch of (lat, lon) pairs to location strings."""
    try:
        results = reverse_geocode.search(coords)
        locations = []
        for geo in results:
            city = geo.get("city", "")
            country = geo.get("country", "")
            if city and country:
                locations.append(f"{city}, {country}")
            else:
                locations.append(city or country)
        return locations
    except Exception:
        logger.debug("Batch reverse geocoding failed", exc_info=True)
        return [""] * len(coords)


PHOTOS_LIBRARY_SOURCE = "__photos_library__"

META_SUFFIX = "-meta"


NOTE_EMB_SUFFIX = "-notes"


@dataclass
class IndexState:
    folders: list[str] = field(default_factory=list)
    photos_library: str | None = None
    enabled_models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    images: list[ImageRecord] = field(default_factory=list)
    notes: list[NoteChunk] = field(default_factory=list)


@dataclass
class SearchResult:
    path: str
    score: float
    preview_thumbnail: str
    model_scores: dict[str, float]
    annotations: list[str] = field(default_factory=list)
    date_taken: str = ""
    location: str = ""
    face_idx: int = -1


# ---------------------------------------------------------------------------
# Date filter parsing
# ---------------------------------------------------------------------------

_DECADE_RE = re.compile(r"\b(\d{4})s\b")
_RANGE_RE = re.compile(r"\b(\d{4})\s*[-–]\s*(\d{4})\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _parse_date_filter(query: str) -> tuple[str, int | None, int | None]:
    """Extract a date range from the query and return the cleaned query.

    Recognizes:
      "2010s"       -> (cleaned, 2010, 2019)
      "2010-2015"   -> (cleaned, 2010, 2015)
      "2010"        -> (cleaned, 2010, 2010)

    Returns (cleaned_query, year_from, year_to). year_from/year_to are None
    when no date expression is found.
    """
    cleaned = query

    # Decade: "2010s" -> 2010-2019
    m = _DECADE_RE.search(cleaned)
    if m:
        base = int(m.group(1))
        cleaned = cleaned[:m.start()] + cleaned[m.end():]
        return cleaned.strip(), base, base + 9

    # Range: "2010-2015"
    m = _RANGE_RE.search(cleaned)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        cleaned = cleaned[:m.start()] + cleaned[m.end():]
        return cleaned.strip(), min(y1, y2), max(y1, y2)

    # Single year: "2010"
    m = _YEAR_RE.search(cleaned)
    if m:
        year = int(m.group(1))
        cleaned = cleaned[:m.start()] + cleaned[m.end():]
        return cleaned.strip(), year, year

    return query, None, None


def _year_from_date_taken(date_taken: str) -> int | None:
    """Extract year from an EXIF date string like '2023:07:15 14:30:00'."""
    if not date_taken:
        return None
    try:
        return int(date_taken[:4])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------

RRF_K = 60
ANNOTATION_WEIGHT = 3.0


def weighted_rrf(
    rankings: dict[str, list[tuple[int, float]]],
    weights: dict[str, float],
    n_images: int,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for model_name, ranked in rankings.items():
        w = weights.get(model_name, 1.0)
        for rank, (idx, _raw_score) in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + w / (RRF_K + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k] if top_k else sorted_items


# ---------------------------------------------------------------------------
# PhotoIndex
# ---------------------------------------------------------------------------


class PhotoIndex:
    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = (cache_dir or CACHE_DIR).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._state = IndexState()
        self._embeddings: dict[str, np.ndarray | None] = {}
        self._meta_embeddings: dict[str, np.ndarray | None] = {}
        self._note_embeddings: dict[str, np.ndarray | None] = {}
        self._models: dict[str, EmbeddingModel] = {}
        self._onnx_text_encoders: dict[str, OnnxTextEncoder] = {}
        self._annotations: dict[str, list[str]] = {}
        self._last_scan: float | None = None

        # Face detection state (loaded lazily from faces.json + faces.npy)
        self._face_records: list[FaceRecord] = []
        self._face_embeddings: np.ndarray | None = None
        self._faces_by_image: dict[str, list[int]] = {}
        self._faces_by_label: dict[str, list[int]] = {}
        self._face_scanned_paths: set[str] = set()  # all images checked, even those without faces
        self._face_engine: FaceEngine | None = None

        # Entity extraction (lazy-initialized)
        self._entity_store: EntityStore | None = None

        self._load_state()

    # -- persistence --

    def _config_path(self) -> Path:
        return self._cache_dir / CONFIG_FILE.name

    def _metadata_path(self) -> Path:
        return self._cache_dir / METADATA_FILE.name

    def _embeddings_dir(self) -> Path:
        d = self._cache_dir / "embeddings"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _embedding_path(self, model_name: str) -> Path:
        return self._embeddings_dir() / f"{model_name}.npy"

    def _meta_embedding_path(self, model_name: str) -> Path:
        return self._embeddings_dir() / f"{model_name}{META_SUFFIX}.npy"

    def _note_embedding_path(self, model_name: str) -> Path:
        return self._embeddings_dir() / f"{model_name}{NOTE_EMB_SUFFIX}.npy"

    def _notes_metadata_path(self) -> Path:
        return self._cache_dir / NOTES_METADATA_FILE.name

    def _annotations_path(self) -> Path:
        return self._cache_dir / ANNOTATIONS_FILE.name

    def _load_annotations(self) -> None:
        path = self._annotations_path()
        if path.exists():
            self._annotations = json.loads(path.read_text())
        else:
            self._annotations = {}

    def _save_annotations(self) -> None:
        self._annotations_path().write_text(json.dumps(self._annotations, indent=2))

    def _load_state(self) -> None:
        cfg = self._config_path()
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text())
                self._state.folders = data.get("folders", [])
                self._state.photos_library = data.get("photos_library")
                self._state.enabled_models = data.get("enabled_models", list(DEFAULT_MODELS))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Corrupt config.json — using defaults")

        meta = self._metadata_path()
        if meta.exists():
            try:
                raw = json.loads(meta.read_text())
                self._state.images = [ImageRecord(**r) for r in raw]
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("Corrupt metadata.json — starting with empty index")
                self._state.images = []

        n_images = len(self._state.images)
        for model_name in self._state.enabled_models:
            npy = self._embedding_path(model_name)
            if npy.exists():
                emb = np.load(str(npy))
                self._embeddings[model_name] = np.array(emb[:n_images]) if len(emb) >= n_images else None
            else:
                self._embeddings[model_name] = None

            meta_npy = self._meta_embedding_path(model_name)
            if meta_npy.exists():
                meta_emb = np.load(str(meta_npy))
                self._meta_embeddings[model_name] = np.array(meta_emb[:n_images]) if len(meta_emb) >= n_images else None
            else:
                self._meta_embeddings[model_name] = None

        # Load notes
        notes_meta = self._notes_metadata_path()
        if notes_meta.exists():
            try:
                raw_notes = json.loads(notes_meta.read_text())
                self._state.notes = [NoteChunk(**n) for n in raw_notes]
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("Corrupt notes_metadata.json — starting with empty notes")
                self._state.notes = []

        n_notes = len(self._state.notes)
        for model_name in self._state.enabled_models:
            note_npy = self._note_embedding_path(model_name)
            if note_npy.exists():
                note_emb = np.load(str(note_npy))
                self._note_embeddings[model_name] = (
                    np.array(note_emb[:n_notes]) if len(note_emb) >= n_notes else None
                )
            else:
                self._note_embeddings[model_name] = None

        self._load_annotations()
        self._load_faces()

    @staticmethod
    def _atomic_write_text(path: Path, data: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(data)
        tmp.rename(path)

    @staticmethod
    def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.stem + "_tmp.npy")
        np.save(str(tmp), arr)
        tmp.rename(path)

    def _save_state(self) -> None:
        # Save embeddings first — metadata.json is the commit point.
        for model_name, emb in self._embeddings.items():
            if emb is not None:
                self._atomic_write_npy(self._embedding_path(model_name), emb)

        for model_name, emb in self._meta_embeddings.items():
            if emb is not None:
                self._atomic_write_npy(self._meta_embedding_path(model_name), emb)

        cfg_data = {
            "folders": self._state.folders,
            "photos_library": self._state.photos_library,
            "enabled_models": self._state.enabled_models,
        }
        self._atomic_write_text(self._config_path(), json.dumps(cfg_data, indent=2))

        # metadata.json written last — acts as commit point.
        meta_data = [asdict(r) for r in self._state.images]
        self._atomic_write_text(self._metadata_path(), json.dumps(meta_data))

        # Save note embeddings and notes_metadata.json
        for model_name, emb in self._note_embeddings.items():
            if emb is not None:
                self._atomic_write_npy(self._note_embedding_path(model_name), emb)

        if self._state.notes:
            notes_data = [asdict(n) for n in self._state.notes]
            self._atomic_write_text(self._notes_metadata_path(), json.dumps(notes_data))

    # -- folder management --

    def add_folder(self, path: str) -> str:
        resolved = str(Path(path).resolve())
        if not Path(resolved).is_dir():
            return f"Not a directory: {resolved}"
        if resolved in self._state.folders:
            return f"Already configured: {resolved}"
        self._state.folders.append(resolved)
        self._save_state()
        return f"Added folder: {resolved}"

    def remove_folder(self, path: str) -> str:
        resolved = str(Path(path).resolve())
        if resolved not in self._state.folders:
            return f"Not configured: {resolved}"
        self._state.folders.remove(resolved)
        self._remove_images_by_folder(resolved)
        self._remove_notes_by_folder(resolved)
        self._save_state()
        return f"Removed folder: {resolved}"

    def list_folders(self) -> list[str]:
        return list(self._state.folders)

    # -- Photos library --

    def connect_photos_library(self, library_path: str | None = None) -> str:
        from photos_library import DEFAULT_LIBRARY

        resolved = str(Path(library_path).resolve()) if library_path else str(DEFAULT_LIBRARY)
        if not Path(resolved).exists():
            return f"Photos library not found: {resolved}"
        self._state.photos_library = resolved
        self._save_state()
        return f"Connected Photos library: {resolved}"

    def disconnect_photos_library(self) -> str:
        if not self._state.photos_library:
            return "No Photos library connected."
        lib = self._state.photos_library
        self._state.photos_library = None
        self._remove_images_by_folder(PHOTOS_LIBRARY_SOURCE)
        self._save_state()
        return f"Disconnected Photos library: {lib}"

    def _remove_images_by_folder(self, folder: str) -> None:
        keep_mask = [r.folder != folder for r in self._state.images]
        self._state.images = [r for r, keep in zip(self._state.images, keep_mask) if keep]
        for model_name in self._state.enabled_models:
            emb = self._embeddings.get(model_name)
            if emb is not None and len(emb) > 0:
                self._embeddings[model_name] = emb[keep_mask]
            meta_emb = self._meta_embeddings.get(model_name)
            if meta_emb is not None and len(meta_emb) > 0:
                self._meta_embeddings[model_name] = meta_emb[keep_mask]
        self._prune_orphaned_annotations()

    def _remove_notes_by_folder(self, folder: str) -> None:
        keep_mask = [n.folder != folder for n in self._state.notes]
        self._state.notes = [n for n, keep in zip(self._state.notes, keep_mask) if keep]
        for model_name in self._state.enabled_models:
            emb = self._note_embeddings.get(model_name)
            if emb is not None and len(emb) > 0:
                self._note_embeddings[model_name] = emb[keep_mask]

    # -- scanning --

    def scan_files(self) -> tuple[list[Path], dict[str, dict], set[str]]:
        """Scan configured folders for image files.

        Returns (paths, metadata_lookup, scanned_folders) where
        scanned_folders is the set of folder strings that were actually
        accessible. Folders on unmounted drives are skipped and excluded
        from the set so refresh() won't treat their images as deleted.
        """
        results = []
        metadata: dict[str, dict] = {}
        scanned_folders: set[str] = set()

        for folder in self._state.folders:
            folder_path = Path(folder)
            if not folder_path.is_dir():
                logger.warning("Folder not accessible (skipping): %s", folder)
                continue
            scanned_folders.add(folder)
            for dirpath, _dirnames, filenames in os.walk(folder_path):
                for fname in filenames:
                    p = Path(dirpath) / fname
                    if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        results.append(p)

        if self._state.photos_library:
            photos_paths, photos_meta = self._scan_photos_library()
            results.extend(photos_paths)
            metadata.update(photos_meta)
            scanned_folders.add(PHOTOS_LIBRARY_SOURCE)

        return results, metadata, scanned_folders

    def _scan_photos_library(self) -> tuple[list[Path], dict[str, dict]]:
        from photos_library import scan_library

        lib_path = Path(self._state.photos_library) if self._state.photos_library else None
        return scan_library(lib_path)

    def _indexed_lookup(self) -> dict[str, tuple[int, ImageRecord]]:
        return {r.path: (i, r) for i, r in enumerate(self._state.images)}

    # -- model management --

    def _onnx_dir(self) -> Path:
        d = self._cache_dir / "onnx"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ensure_model(self, model_name: str) -> EmbeddingModel:
        if model_name in self._models and self._models[model_name].loaded:
            return self._models[model_name]
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        model = AVAILABLE_MODELS[model_name]
        device = get_device()
        model.load(device)
        self._models[model_name] = model

        # Export text encoder to ONNX for fast search on next startup
        if isinstance(model, OpenCLIPModel):
            export_text_encoder(model, self._onnx_dir())

        return model

    def _ensure_text_encoder(self, model_name: str):
        """Get a text encoder for search. Prefers ONNX over full PyTorch model."""
        enc = self._onnx_text_encoders.get(model_name)
        if enc is not None and enc.loaded:
            return enc

        enc = load_text_encoder(model_name, self._onnx_dir())
        if enc is not None:
            enc.load()
            self._onnx_text_encoders[model_name] = enc
            return enc

        # Fall back to full model
        return self._ensure_model(model_name)

    def _text_capable_models(self) -> list[str]:
        return [
            m for m in self._state.enabled_models
            if AVAILABLE_MODELS.get(m, None) and AVAILABLE_MODELS[m].supports_text
        ]

    def _loaded_models(self) -> dict[str, EmbeddingModel]:
        return {
            name: self._ensure_model(name)
            for name in self._state.enabled_models
        }

    # -- refresh (incremental update) --

    def refresh(self, progress_callback: ProgressCallback | None = None) -> dict:
        """Incremental refresh: append-only. Never deletes data.

        Scans configured folders, finds files not yet in the index, and
        processes them. Existing entries are never removed — use
        prune_deleted() for explicit, user-initiated cleanup.
        """
        current_files, metadata_lookup, _scanned = self.scan_files()
        indexed = self._indexed_lookup()
        current_by_path: dict[str, Path] = {str(p.resolve()): p for p in current_files}

        new_paths = [
            p for path_str, p in current_by_path.items()
            if path_str not in indexed
        ]

        if not new_paths:
            self._last_scan = time.time()
            note_stats = self._refresh_notes()
            return {
                "new": 0,
                "total": len(self._state.images),
                "notes_new": note_stats.get("new", 0),
                "total_notes": note_stats.get("total", len(self._state.notes)),
            }

        models = self._loaded_models()
        checkpoints = run_pipeline(
            paths=new_paths,
            folder_lookup=self._find_folder,
            models=models,
            progress_callback=progress_callback,
            metadata_lookup=metadata_lookup or None,
        )

        total_added = 0
        for cp in checkpoints:
            self._state.images.extend(cp.records)

            for model_name, emb in cp.embeddings.items():
                existing = self._embeddings.get(model_name)
                if existing is not None and len(existing) > 0:
                    self._embeddings[model_name] = np.vstack([existing, emb])
                else:
                    self._embeddings[model_name] = emb

            for model_name, meta_emb in cp.meta_embeddings.items():
                existing = self._meta_embeddings.get(model_name)
                if existing is not None and len(existing) > 0:
                    self._meta_embeddings[model_name] = np.vstack([existing, meta_emb])
                else:
                    self._meta_embeddings[model_name] = meta_emb

            total_added += len(cp.records)
            try:
                self._save_state()
            except OSError:
                logger.warning("Checkpoint save failed, will retry next checkpoint", exc_info=True)

        note_stats = self._refresh_notes(models)

        self._last_scan = time.time()
        self._save_state()

        return {
            "new": total_added,
            "total": len(self._state.images),
            "notes_new": note_stats.get("new", 0),
            "total_notes": note_stats.get("total", len(self._state.notes)),
        }

    def _find_folder(self, path: Path) -> str:
        resolved = str(path.resolve())
        if self._state.photos_library and resolved.startswith(self._state.photos_library):
            return PHOTOS_LIBRARY_SOURCE
        for folder in self._state.folders:
            if resolved.startswith(folder):
                return folder
        return str(path.parent)

    # -- notes --

    def _refresh_notes(self, models: dict[str, EmbeddingModel] | None = None) -> dict:
        """Scan folders for text files, chunk, and embed. Append-only."""
        current_files = scan_notes(self._state.folders)
        current_by_path: dict[str, Path] = {str(p.resolve()): p for p in current_files}

        indexed_paths = {n.path for n in self._state.notes}

        new_file_paths = [
            p for path_str, p in current_by_path.items()
            if path_str not in indexed_paths
        ]

        if not new_file_paths:
            return {"new": 0, "total": len(self._state.notes)}

        new_chunks: list[NoteChunk] = []
        for p in new_file_paths:
            folder = self._find_folder(p)
            new_chunks.extend(chunk_file(p, folder))

        if not new_chunks:
            return {"new": 0, "total": len(self._state.notes)}

        if models is None:
            models = self._loaded_models()

        text_models = {
            name: m for name, m in models.items() if m.supports_text
        }

        texts = [c.text for c in new_chunks]
        for model_name, model in text_models.items():
            emb = model.encode_texts(texts)
            existing = self._note_embeddings.get(model_name)
            if existing is not None and len(existing) > 0:
                self._note_embeddings[model_name] = np.vstack([existing, emb])
            else:
                self._note_embeddings[model_name] = emb

        n_new = len(new_chunks)
        self._state.notes.extend(new_chunks)
        self._save_state()

        # Run entity extraction on new chunks
        if n_new > 0:
            try:
                self.extract_entities()
            except Exception:
                logger.warning("Entity extraction failed during notes refresh", exc_info=True)

        logger.info("Notes refresh: +%d new, %d total", n_new, len(self._state.notes))
        return {"new": n_new, "total": len(self._state.notes)}

    # -- entity extraction --

    def _entities_db_path(self) -> Path:
        return self._cache_dir / ENTITIES_DB.name

    def get_entity_store(self) -> EntityStore:
        if self._entity_store is None:
            self._entity_store = EntityStore(self._entities_db_path())
        return self._entity_store

    def extract_entities(self, progress_callback=None) -> dict:
        """Run NER on all note chunks. Incremental - skips already-processed."""
        notes = self._state.notes
        if not notes:
            return {"chunks_processed": 0, "chunks_skipped": 0, "mentions_added": 0}

        documents = [n.text for n in notes]
        metadatas = [
            {
                "rel_path": n.rel_path,
                "chunk_index": n.chunk_index,
                "mtime": n.mtime,
                "file_path": n.path,
                "start_line": n.start_line,
                "end_line": n.end_line,
            }
            for n in notes
        ]

        store = self.get_entity_store()
        return store.extract_from_chunks(documents, metadatas, progress_callback)

    def search_notes(self, query: str, top_k: int = 10) -> list[NoteSearchResult]:
        """Search notes by encoding query with CLIP/SigLIP text encoder."""
        notes = self._state.notes
        note_embeddings = self._note_embeddings

        if not notes:
            return []

        text_models = self._text_capable_models()
        if not text_models:
            return []

        n = len(notes)
        # Fuse scores across models (simple average)
        fused_scores = np.zeros(n, dtype=np.float32)
        n_models = 0

        for model_name in text_models:
            encoder = self._ensure_text_encoder(model_name)
            query_vec = encoder.encode_text(query)

            emb = note_embeddings.get(model_name)
            if emb is None or len(emb) < n:
                continue

            scores = emb[:n] @ query_vec
            fused_scores += scores
            n_models += 1

        if n_models == 0:
            return []

        fused_scores /= n_models
        ranked = np.argsort(fused_scores)[::-1]

        # Apply per-file diversity limit
        results: list[NoteSearchResult] = []
        file_counts: dict[str, int] = {}
        for idx in ranked:
            if len(results) >= top_k:
                break
            note = notes[int(idx)]
            count = file_counts.get(note.path, 0)
            if count >= NOTE_MAX_PER_FILE:
                continue
            file_counts[note.path] = count + 1
            results.append(NoteSearchResult(
                text=note.text,
                rel_path=note.rel_path,
                start_line=note.start_line,
                end_line=note.end_line,
                score=float(fused_scores[idx]),
            ))

        return results

    # -- annotations --

    def annotate(self, path: str, text: str) -> str:
        resolved = str(Path(path).resolve())
        known_paths = {r.path for r in self._state.images}
        if resolved not in known_paths:
            return f"Image not in index: {resolved}"
        text = text.strip()
        if not text:
            return "Annotation text cannot be empty."
        if resolved not in self._annotations:
            self._annotations[resolved] = []
        if text in self._annotations[resolved]:
            return f"Annotation already exists for {Path(resolved).name}"
        self._annotations[resolved].append(text)
        self._save_annotations()
        return f"Annotated {Path(resolved).name}: {text}"

    def get_annotations(self, path: str) -> list[str]:
        resolved = str(Path(path).resolve())
        return list(self._annotations.get(resolved, []))

    def remove_annotation(self, path: str, text: str) -> str:
        resolved = str(Path(path).resolve())
        annotations = self._annotations.get(resolved, [])
        if text not in annotations:
            return f"Annotation not found on {Path(resolved).name}"
        annotations.remove(text)
        if not annotations:
            del self._annotations[resolved]
        self._save_annotations()
        return f"Removed annotation from {Path(resolved).name}"

    def pending_tasks(self) -> list[dict]:
        """Return all annotations that start with '@task '."""
        tasks = []
        for path, annotations in self._annotations.items():
            for text in annotations:
                if text.startswith("@task "):
                    tasks.append({"path": path, "task": text})
        return tasks

    def _prune_orphaned_annotations(self) -> None:
        remaining_paths = {r.path for r in self._state.images}
        orphans = [p for p in self._annotations if p not in remaining_paths]
        for p in orphans:
            del self._annotations[p]
        if orphans:
            self._save_annotations()

    def _rank_by_annotations(
        self,
        query: str,
        images: list[ImageRecord],
        annotations: dict[str, list[str]],
    ) -> list[tuple[int, float]]:
        scored = []
        for idx, record in enumerate(images):
            anns = annotations.get(record.path, [])
            if not anns:
                continue
            score = score_annotations(query, anns)
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _rank_by_metadata_keywords(
        query: str,
        images: list[ImageRecord],
    ) -> list[tuple[int, float]]:
        """Keyword match on metadata_text (filenames, people, keywords, etc.)."""
        scored = []
        for idx, record in enumerate(images):
            if not record.metadata_text:
                continue
            score = score_annotations(query, [record.metadata_text])
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # -- search --

    def _build_results(
        self,
        rankings: dict[str, list[tuple[int, float]]],
        raw_scores: dict[str, np.ndarray],
        images: list[ImageRecord],
        annotations: dict[str, list[str]],
        n: int,
        top_k: int | None = None,
        score_ratio: float = 0.0,
    ) -> list[SearchResult]:
        weights = {m: 1.0 for m in rankings}
        if "annotations" in weights:
            weights["annotations"] = ANNOTATION_WEIGHT
        if "metadata_keywords" in weights:
            weights["metadata_keywords"] = ANNOTATION_WEIGHT

        fused = weighted_rrf(rankings, weights, n, top_k)

        # Compute per-image max visual cosine similarity for threshold filtering.
        # We threshold on score range: cutoff = worst + (best - worst) * ratio.
        # This keeps results in the top (1 - ratio) fraction of the score range.
        visual_keys = [k for k in raw_scores if not k.endswith(META_SUFFIX)]
        cutoff = 0.0
        if score_ratio > 0.0 and visual_keys and fused:
            per_image_max = np.array([
                max(float(raw_scores[k][idx]) for k in visual_keys)
                for idx, _ in fused
            ])
            best_visual = float(per_image_max.max())
            worst_visual = float(per_image_max.min())
            cutoff = worst_visual + (best_visual - worst_visual) * score_ratio

        results = []
        coords_batch = []
        coord_result_indices = []

        for i, (idx, fused_score) in enumerate(fused):
            if cutoff > 0.0:
                max_visual = max(float(raw_scores[k][idx]) for k in visual_keys)
                if max_visual < cutoff:
                    continue

            record = images[idx]
            per_model = {}
            for mn, scores_arr in raw_scores.items():
                per_model[mn] = round(float(scores_arr[idx]), 4)

            preview_path = CACHE_DIR / "thumbnails" / "preview" / f"{record.thumb_key}.webp"
            results.append(SearchResult(
                path=record.path,
                score=round(fused_score, 4),
                preview_thumbnail=str(preview_path),
                model_scores=per_model,
                annotations=list(annotations.get(record.path, [])),
                date_taken=record.date_taken,
            ))

            if record.latitude is not None and record.longitude is not None:
                coords_batch.append((record.latitude, record.longitude))
                coord_result_indices.append(len(results) - 1)

        if coords_batch:
            locations = _batch_reverse_geocode(coords_batch)
            for i, loc in zip(coord_result_indices, locations):
                results[i].location = loc

        return results

    def _filter_by_date(
        self,
        results: list[SearchResult],
        year_from: int,
        year_to: int,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        filtered = []
        for r in results:
            y = _year_from_date_taken(r.date_taken)
            if y is not None and year_from <= y <= year_to:
                filtered.append(r)
                if top_k and len(filtered) >= top_k:
                    break
        return filtered

    def search_progressive(self, query: str, top_k: int | None = None, score_ratio: float | None = None):
        """Generator yielding result lists as each ranking signal completes.

        Yields after the first model scores (fast initial results), then
        yields the fully fused results after all models and keyword signals.
        Supports date filtering when the query contains a year expression.
        """
        visual_query, year_from, year_to = _parse_date_filter(query)
        has_date_filter = year_from is not None

        # Use the full query for keyword/annotation matching, cleaned for embeddings
        embedding_query = visual_query if has_date_filter else query
        # Fetch more candidates when filtering so we don't end up with too few
        fetch_k = top_k * 4 if (has_date_filter and top_k) else top_k

        images = list(self._state.images)
        embeddings = {k: v for k, v in self._embeddings.items()}
        meta_embeddings = {k: v for k, v in self._meta_embeddings.items()}
        annotations = dict(self._annotations)

        if not images:
            return

        text_models = self._text_capable_models()
        if not text_models:
            return

        n = len(images)
        rankings: dict[str, list[tuple[int, float]]] = {}
        raw_scores: dict[str, np.ndarray] = {}

        for i, model_name in enumerate(text_models):
            encoder = self._ensure_text_encoder(model_name)
            query_vec = encoder.encode_text(embedding_query)

            emb = embeddings.get(model_name)
            if emb is not None and len(emb) >= n:
                emb = emb[:n]
                scores = emb @ query_vec
                raw_scores[model_name] = scores
                ranked_indices = np.argsort(scores)[::-1]
                rankings[model_name] = [(int(j), float(scores[j])) for j in ranked_indices]

            meta_emb = meta_embeddings.get(model_name)
            if meta_emb is not None and len(meta_emb) >= n:
                meta_emb = meta_emb[:n]
                meta_scores = meta_emb @ query_vec
                meta_key = f"{model_name}{META_SUFFIX}"
                raw_scores[meta_key] = meta_scores
                ranked_meta = np.argsort(meta_scores)[::-1]
                rankings[meta_key] = [(int(j), float(meta_scores[j])) for j in ranked_meta]

            # Yield after first model for instant response
            if i == 0 and rankings:
                ratio = score_ratio if score_ratio is not None else SEARCH_SCORE_RATIO
                results = self._build_results(rankings, raw_scores, images, annotations, n, fetch_k, score_ratio=ratio)
                if has_date_filter:
                    results = self._filter_by_date(results, year_from, year_to, top_k)
                yield results

        # Add keyword signals (use original query for keyword matching)
        ann_ranking = self._rank_by_annotations(query, images, annotations)
        if ann_ranking:
            rankings["annotations"] = ann_ranking

        meta_kw_ranking = self._rank_by_metadata_keywords(query, images)
        if meta_kw_ranking:
            rankings["metadata_keywords"] = meta_kw_ranking

        if rankings:
            ratio = score_ratio if score_ratio is not None else SEARCH_SCORE_RATIO
            results = self._build_results(rankings, raw_scores, images, annotations, n, fetch_k, score_ratio=ratio)
            if has_date_filter:
                results = self._filter_by_date(results, year_from, year_to, top_k)
            yield results

    def search(self, query: str, top_k: int | None = None, score_ratio: float | None = None) -> list[SearchResult]:
        results: list[SearchResult] = []
        for batch in self.search_progressive(query, top_k, score_ratio=score_ratio):
            results = batch
        return results

    def find_similar(self, path: str, top_k: int | None = None, score_ratio: float | None = None) -> list[SearchResult]:
        """Find images visually similar to the given image.

        Looks up the image's pre-computed embeddings and computes cosine
        similarity against all other indexed images. No model loading needed.
        """
        resolved = str(Path(path).resolve())
        indexed = self._indexed_lookup()
        if resolved not in indexed:
            return []

        query_idx, _record = indexed[resolved]
        images = list(self._state.images)
        embeddings = {k: v for k, v in self._embeddings.items()}
        annotations = dict(self._annotations)
        n = len(images)

        rankings: dict[str, list[tuple[int, float]]] = {}
        raw_scores: dict[str, np.ndarray] = {}

        for model_name in self._state.enabled_models:
            emb = embeddings.get(model_name)
            if emb is None or len(emb) < n:
                continue
            emb = emb[:n]
            query_vec = emb[query_idx]
            scores = emb @ query_vec
            # Exclude the query image itself
            scores[query_idx] = -1.0
            raw_scores[model_name] = scores
            ranked_indices = np.argsort(scores)[::-1]
            rankings[model_name] = [(int(j), float(scores[j])) for j in ranked_indices]

        if not rankings:
            return []

        ratio = score_ratio if score_ratio is not None else SEARCH_SCORE_RATIO
        return self._build_results(rankings, raw_scores, images, annotations, n, top_k, score_ratio=ratio)

    # -- prune / reindex --

    def prune_deleted(self) -> dict:
        """Remove index entries for files that no longer exist on disk.

        This is the ONLY way to remove images from the index (besides
        full_reindex). Must be called explicitly — never called automatically.
        """
        remove_set: set[int] = set()
        for i, rec in enumerate(self._state.images):
            if not Path(rec.path).exists():
                remove_set.add(i)

        if not remove_set:
            return {"pruned": 0, "total": len(self._state.images)}

        keep_mask = [i not in remove_set for i in range(len(self._state.images))]
        self._state.images = [
            r for r, keep in zip(self._state.images, keep_mask) if keep
        ]
        for model_name in self._state.enabled_models:
            emb = self._embeddings.get(model_name)
            if emb is not None and len(emb) > 0:
                self._embeddings[model_name] = emb[keep_mask]
            meta_emb = self._meta_embeddings.get(model_name)
            if meta_emb is not None and len(meta_emb) > 0:
                self._meta_embeddings[model_name] = meta_emb[keep_mask]

        self._prune_orphaned_annotations()
        self._save_state()

        logger.info("Pruned %d deleted images, %d remain", len(remove_set), len(self._state.images))
        return {"pruned": len(remove_set), "total": len(self._state.images)}

    def reindex_notes(self, progress_callback: ProgressCallback | None = None) -> dict:
        """Drop and rebuild only the notes index. Images are untouched."""
        start = time.time()
        self._state.notes.clear()
        self._note_embeddings.clear()
        self._save_state()
        note_stats = self._refresh_notes()
        elapsed = time.time() - start
        return {
            "notes": note_stats.get("total", 0),
            "elapsed": elapsed,
        }

    def full_reindex(
        self,
        progress_callback: ProgressCallback | None = None,
        *,
        confirm: bool = False,
    ) -> dict:
        """Drop and rebuild the entire index (images + notes + embeddings).

        Requires confirm=True to prevent accidental invocation. Without it,
        raises ValueError — this operation takes hours on large libraries.
        """
        if not confirm:
            raise ValueError(
                "full_reindex requires confirm=True. "
                "This operation drops all embeddings and re-processes every image."
            )
        start = time.time()
        self._state.images.clear()
        self._state.notes.clear()
        self._embeddings.clear()
        self._meta_embeddings.clear()
        self._note_embeddings.clear()
        stats = self.refresh(progress_callback=progress_callback)
        elapsed = time.time() - start
        return {
            "images": stats["total"],
            "notes": stats.get("total_notes", 0),
            "models": self._state.enabled_models,
            "elapsed": elapsed,
        }

    # -- status --

    def _cache_size_bytes(self) -> int:
        """Total size of the cache directory in bytes."""
        total = 0
        try:
            for entry in self._cache_dir.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except OSError:
            pass
        return total

    def status(self) -> dict:
        return {
            "folders": self._state.folders,
            "photos_library": self._state.photos_library,
            "enabled_models": self._state.enabled_models,
            "total_images": len(self._state.images),
            "annotated_images": len(self._annotations),
            "embeddings": {
                m: (len(e) if e is not None else 0)
                for m, e in self._embeddings.items()
            },
            "total_notes": len(self._state.notes),
            "total_faces": len(self._face_records),
            "labeled_faces": sum(1 for r in self._face_records if r.label),
            "people": len(self._faces_by_label),
            "last_scan": self._last_scan,
            "cache_size_bytes": self._cache_size_bytes(),
        }

    # -- face detection and recognition --

    def _faces_path(self) -> Path:
        return self._cache_dir / FACES_FILE.name

    def _face_embeddings_path(self) -> Path:
        return self._embeddings_dir() / "faces.npy"

    def _load_faces(self) -> None:
        path = self._faces_path()
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                # Support both old format (plain list) and new format (dict)
                if isinstance(raw, list):
                    records_raw = raw
                    scanned_raw = []
                else:
                    records_raw = raw.get("records", [])
                    # Backward compat: old faces.json may contain "scanned"
                    scanned_raw = raw.get("scanned", [])
                self._face_records = [
                    FaceRecord(
                        image_path=r["image_path"],
                        face_idx=r["face_idx"],
                        bbox=tuple(r["bbox"]),
                        confidence=r["confidence"],
                        label=r.get("label", ""),
                    )
                    for r in records_raw
                ]
                if scanned_raw:
                    self._face_scanned_paths.update(scanned_raw)
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("Corrupt faces.json -- starting with empty face index")
                self._face_records = []
                self._face_scanned_paths = set()

        # Load scanned paths from dedicated file
        scanned_file = self._faces_scanned_path()
        if scanned_file.exists():
            try:
                text = scanned_file.read_text()
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        self._face_scanned_paths.add(line)
            except OSError:
                logger.debug("Failed to read faces_scanned.txt")

        n = len(self._face_records)
        npy = self._face_embeddings_path()
        if npy.exists() and n > 0:
            emb = np.load(str(npy))
            self._face_embeddings = np.array(emb[:n]) if len(emb) >= n else None
        else:
            self._face_embeddings = None

        self._rebuild_face_lookups()

    @staticmethod
    def _face_record_to_dict(rec) -> dict:
        d = asdict(rec)
        d["bbox"] = tuple(float(v) for v in d["bbox"])
        d["confidence"] = float(d["confidence"])
        return d

    def _faces_scanned_path(self) -> Path:
        return self._cache_dir / FACES_SCANNED_FILE.name

    def _save_faces(self) -> None:
        data = {
            "records": [self._face_record_to_dict(r) for r in self._face_records],
        }
        self._atomic_write_text(self._faces_path(), json.dumps(data))
        if self._face_embeddings is not None:
            self._atomic_write_npy(self._face_embeddings_path(), self._face_embeddings)
        self._atomic_write_text(
            self._faces_scanned_path(),
            "\n".join(sorted(self._face_scanned_paths)) + "\n" if self._face_scanned_paths else "",
        )

    def _rebuild_face_lookups(self) -> None:
        self._faces_by_image = {}
        self._faces_by_label = {}
        for i, rec in enumerate(self._face_records):
            self._faces_by_image.setdefault(rec.image_path, []).append(i)
            self._face_scanned_paths.add(rec.image_path)
            if rec.label:
                self._faces_by_label.setdefault(rec.label, []).append(i)

    def _ensure_face_engine(self) -> FaceEngine:
        if self._face_engine is None or not self._face_engine.loaded:
            self._face_engine = FaceEngine()
            self._face_engine.load()
        return self._face_engine

    def _prefilter_people(
        self, candidates: list[tuple[int, "ImageRecord"]], threshold: float = 0.105
    ) -> list[tuple[int, "ImageRecord"]]:
        """Use fused CLIP+SigLIP scores to keep images likely containing people.

        Averages cosine similarity to "a photo of a person" across all
        text-capable models. Pure numpy on pre-computed embeddings — milliseconds.
        """
        text_models = self._text_capable_models()
        if not text_models:
            return candidates

        n = len(self._state.images)
        query = "a photo of a person"
        fused = np.zeros(n, dtype=np.float32)
        n_models = 0

        for model_name in text_models:
            emb_matrix = self._embeddings.get(model_name)
            if emb_matrix is None or len(emb_matrix) < n:
                continue
            encoder = self._ensure_text_encoder(model_name)
            query_vec = encoder.encode_text(query)
            fused += emb_matrix[:n] @ query_vec
            n_models += 1

        if n_models == 0:
            return candidates
        fused /= n_models

        kept = []
        for global_idx, image_rec in candidates:
            if fused[global_idx] >= threshold:
                kept.append((global_idx, image_rec))

        logger.info(
            "Face pre-filter: %d/%d candidates likely contain people (threshold=%.2f, models=%d)",
            len(kept), len(candidates), threshold, n_models,
        )
        return kept

    def refresh_faces(self, progress_callback=None) -> dict:
        """Detect faces in images not yet scanned. Append-only.

        Uses CLIP embeddings to pre-filter images likely containing people
        before running the heavier face detection model.

        Memory strategy: pre-allocate an embedding buffer sized for the
        expected faces.  Each new embedding is written directly into the
        buffer (O(1) per face, zero copies).  A single save happens at
        the end -- no intermediate full-file rewrites.
        """
        faces_before = len(self._face_records)
        already_scanned = self._face_scanned_paths | set(self._faces_by_image)
        all_pending = [
            (i, r) for i, r in enumerate(self._state.images)
            if r.path not in already_scanned
        ]

        if not all_pending:
            return {
                "new_faces": 0,
                "images_scanned": 0,
                "images_skipped": 0,
                "total_faces": len(self._face_records),
            }

        # Pre-filter: only run face detection on images likely to contain people
        candidates = self._prefilter_people(all_pending)
        skipped = len(all_pending) - len(candidates)

        # Pre-allocate embedding buffer: existing faces + ~2 per candidate
        existing_count = len(self._face_records)
        estimated_new = max(len(candidates) * 2, 64)
        capacity = existing_count + estimated_new
        buffer = np.empty((capacity, 512), dtype=np.float32)
        if self._face_embeddings is not None:
            buffer[:existing_count] = self._face_embeddings
        write_pos = existing_count

        # Point live queries at the buffer view
        self._face_embeddings = buffer[:write_pos]

        engine = self._ensure_face_engine()
        t0 = time.time()

        for i, (global_idx, image_rec) in enumerate(candidates):
            if progress_callback:
                elapsed = time.time() - t0
                rate = f"{(i + 1) / elapsed:.1f} faces/sec" if elapsed > 0 else ""
                progress_callback(i + 1, len(candidates), rate)

            # Use preview thumbnail for speed (800px, already generated)
            thumb_path = str(
                self._cache_dir
                / "thumbnails"
                / "preview"
                / f"{image_rec.thumb_key}.webp"
            )
            source = thumb_path if Path(thumb_path).exists() else image_rec.path

            try:
                results = engine.detect_and_embed(source)
                for face_rec, emb in results:
                    face_rec.image_path = image_rec.path
                    self._face_records.append(face_rec)

                    # Grow buffer if needed (doubling strategy)
                    if write_pos >= capacity:
                        capacity *= 2
                        new_buf = np.empty((capacity, 512), dtype=np.float32)
                        new_buf[:write_pos] = buffer[:write_pos]
                        buffer = new_buf

                    buffer[write_pos] = emb.astype(np.float32)
                    write_pos += 1

                    # Update in-memory lookups incrementally
                    idx = len(self._face_records) - 1
                    self._faces_by_image.setdefault(face_rec.image_path, []).append(idx)
                    if face_rec.label:
                        self._faces_by_label.setdefault(face_rec.label, []).append(idx)

                    # Keep live query view current
                    self._face_embeddings = buffer[:write_pos]
            except Exception:
                logger.debug("Face detection failed: %s", image_rec.path, exc_info=True)

            self._face_scanned_paths.add(image_rec.path)

        # Final: trim buffer to actual size and save once
        self._face_embeddings = buffer[:write_pos].copy()
        self._save_faces()

        return {
            "new_faces": len(self._face_records) - faces_before,
            "images_scanned": len(candidates),
            "images_skipped": skipped,
            "total_faces": len(self._face_records),
        }

    def get_faces(self, path: str) -> list[dict]:
        """Return face records for a given image."""
        resolved = str(Path(path).resolve())
        indices = self._faces_by_image.get(resolved, [])
        return [asdict(self._face_records[i]) for i in indices]

    def label_face(self, path: str, face_idx: int, label: str) -> str:
        """Assign a person name to a detected face."""
        resolved = str(Path(path).resolve())
        indices = self._faces_by_image.get(resolved, [])
        for i in indices:
            if self._face_records[i].face_idx == face_idx:
                self._face_records[i].label = label.strip()
                self._rebuild_face_lookups()
                self._save_faces()
                # Also add @person annotation for text search
                self.annotate(resolved, f"@person {label.strip()}")
                return f"Labeled face {face_idx} as '{label.strip()}'"
        return f"Face {face_idx} not found in {Path(resolved).name}"

    def find_person(
        self, path: str, face_idx: int = 0, top_k: int | None = None
    ) -> list[SearchResult]:
        """Find images containing the same person as a given face.

        Uses pre-computed ArcFace embeddings for cosine similarity.
        Returns SearchResult list (same format as find_similar).
        """
        resolved = str(Path(path).resolve())
        indices = self._faces_by_image.get(resolved, [])

        query_global_idx = None
        for i in indices:
            if self._face_records[i].face_idx == face_idx:
                query_global_idx = i
                break
        if query_global_idx is None or self._face_embeddings is None:
            return []

        query_vec = self._face_embeddings[query_global_idx]
        scores = self._face_embeddings @ query_vec

        # Rank all faces, deduplicate by image, filter by threshold
        ranked = np.argsort(scores)[::-1]
        images = list(self._state.images)
        indexed = self._indexed_lookup()
        annotations = dict(self._annotations)

        seen_images: set[str] = set()
        results: list[SearchResult] = []
        coords_batch: list[tuple[float, float]] = []
        coord_indices: list[int] = []

        for idx in ranked:
            idx = int(idx)
            if idx == query_global_idx:
                continue
            sim = float(scores[idx])
            if sim < FACE_SIMILARITY_THRESHOLD:
                break

            rec = self._face_records[idx]
            if rec.image_path in seen_images:
                continue
            seen_images.add(rec.image_path)

            lookup = indexed.get(rec.image_path)
            if lookup is None:
                continue
            img_idx, image_rec = lookup

            preview_path = (
                self._cache_dir
                / "thumbnails"
                / "preview"
                / f"{image_rec.thumb_key}.webp"
            )
            result = SearchResult(
                path=rec.image_path,
                score=round(sim, 4),
                preview_thumbnail=str(preview_path),
                model_scores={"face": round(sim, 4)},
                annotations=list(annotations.get(rec.image_path, [])),
                date_taken=image_rec.date_taken,
                face_idx=rec.face_idx,
            )
            ri = len(results)
            results.append(result)

            if image_rec.latitude is not None and image_rec.longitude is not None:
                coords_batch.append((image_rec.latitude, image_rec.longitude))
                coord_indices.append(ri)

            if top_k and len(results) >= top_k:
                break

        if coords_batch:
            locations = _batch_reverse_geocode(coords_batch)
            for i, loc in zip(coord_indices, locations):
                results[i].location = loc

        return results

    def batch_label_faces(self, matches: list[dict], label: str) -> dict:
        """Label multiple faces at once. Each match needs image_path and face_idx."""
        labeled = 0
        label = label.strip()
        for match in matches:
            resolved = str(Path(match["image_path"]).resolve())
            face_idx = match["face_idx"]
            indices = self._faces_by_image.get(resolved, [])
            for i in indices:
                if self._face_records[i].face_idx == face_idx:
                    if self._face_records[i].label != label:
                        self._face_records[i].label = label
                        self.annotate(resolved, f"@person {label}")
                        labeled += 1
                    break

        self._rebuild_face_lookups()
        self._save_faces()
        return {"labeled": labeled, "label": label}

    def get_people(self) -> list[dict]:
        """Return all known people with face and image counts."""
        people = {}
        for label, indices in self._faces_by_label.items():
            image_paths = {self._face_records[i].image_path for i in indices}
            people[label] = {
                "name": label,
                "face_count": len(indices),
                "image_count": len(image_paths),
            }
        return sorted(people.values(), key=lambda p: p["image_count"], reverse=True)
