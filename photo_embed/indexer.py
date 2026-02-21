import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from annotations import score_annotations
from config import (
    ANNOTATIONS_FILE,
    CACHE_DIR,
    CONFIG_FILE,
    DEFAULT_MODELS,
    METADATA_FILE,
    SUPPORTED_EXTENSIONS,
)
from models import AVAILABLE_MODELS, EmbeddingModel, get_device
from pipeline import Checkpoint, ImageRecord, ProgressCallback, run_pipeline

logger = logging.getLogger(__name__)


PHOTOS_LIBRARY_SOURCE = "__photos_library__"

META_SUFFIX = "-meta"


@dataclass
class IndexState:
    folders: list[str] = field(default_factory=list)
    photos_library: str | None = None
    enabled_models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    images: list[ImageRecord] = field(default_factory=list)


@dataclass
class SearchResult:
    path: str
    score: float
    preview_thumbnail: str
    model_scores: dict[str, float]
    annotations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------

RRF_K = 60
ANNOTATION_WEIGHT = 3.0


def weighted_rrf(
    rankings: dict[str, list[tuple[int, float]]],
    weights: dict[str, float],
    n_images: int,
    top_k: int,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for model_name, ranked in rankings.items():
        w = weights.get(model_name, 1.0)
        for rank, (idx, _raw_score) in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + w / (RRF_K + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]


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
        self._models: dict[str, EmbeddingModel] = {}
        self._annotations: dict[str, list[str]] = {}
        self._last_scan: float | None = None
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
                # Trim to match metadata record count (commit point)
                self._embeddings[model_name] = emb[:n_images] if len(emb) >= n_images else None
            else:
                self._embeddings[model_name] = None

            meta_npy = self._meta_embedding_path(model_name)
            if meta_npy.exists():
                meta_emb = np.load(str(meta_npy))
                self._meta_embeddings[model_name] = meta_emb[:n_images] if len(meta_emb) >= n_images else None
            else:
                self._meta_embeddings[model_name] = None

        self._load_annotations()

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

    # -- scanning --

    def scan_files(self) -> tuple[list[Path], dict[str, dict]]:
        results = []
        metadata: dict[str, dict] = {}

        for folder in self._state.folders:
            folder_path = Path(folder)
            if not folder_path.is_dir():
                logger.warning("Folder not accessible: %s", folder)
                continue
            for dirpath, _dirnames, filenames in os.walk(folder_path):
                for fname in filenames:
                    p = Path(dirpath) / fname
                    if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        results.append(p)

        if self._state.photos_library:
            photos_paths, photos_meta = self._scan_photos_library()
            results.extend(photos_paths)
            metadata.update(photos_meta)

        return results, metadata

    def _scan_photos_library(self) -> tuple[list[Path], dict[str, dict]]:
        from photos_library import scan_library

        lib_path = Path(self._state.photos_library) if self._state.photos_library else None
        return scan_library(lib_path)

    def _indexed_lookup(self) -> dict[str, tuple[int, ImageRecord]]:
        return {r.path: (i, r) for i, r in enumerate(self._state.images)}

    # -- model management --

    def _ensure_model(self, model_name: str) -> EmbeddingModel:
        if model_name in self._models and self._models[model_name].loaded:
            return self._models[model_name]
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        model = AVAILABLE_MODELS[model_name]
        device = get_device()
        model.load(device)
        self._models[model_name] = model
        return model

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
        current_files, metadata_lookup = self.scan_files()
        indexed = self._indexed_lookup()

        current_by_path: dict[str, Path] = {str(p.resolve()): p for p in current_files}

        new_paths: list[Path] = []
        changed_indices: list[int] = []
        for path_str, p in current_by_path.items():
            mtime = os.path.getmtime(p)
            if path_str not in indexed:
                new_paths.append(p)
            else:
                idx, record = indexed[path_str]
                if mtime > record.mtime:
                    changed_indices.append(idx)
                    new_paths.append(p)

        deleted_indices = [
            idx for path_str, (idx, _rec) in indexed.items()
            if path_str not in current_by_path
        ]

        if not new_paths and not changed_indices and not deleted_indices:
            self._last_scan = time.time()
            return {"new": 0, "changed": 0, "deleted": 0, "total": len(self._state.images)}

        # Remove changed and deleted entries
        remove_set = set(changed_indices) | set(deleted_indices)
        if remove_set:
            keep_mask = [i not in remove_set for i in range(len(self._state.images))]
            self._state.images = [r for r, keep in zip(self._state.images, keep_mask) if keep]

            if deleted_indices:
                self._prune_orphaned_annotations()

            for model_name in self._state.enabled_models:
                emb = self._embeddings.get(model_name)
                if emb is not None and len(emb) > 0:
                    self._embeddings[model_name] = emb[keep_mask]
                meta_emb = self._meta_embeddings.get(model_name)
                if meta_emb is not None and len(meta_emb) > 0:
                    self._meta_embeddings[model_name] = meta_emb[keep_mask]

            self._save_state()

        # Run the streaming pipeline for new/changed images
        total_added = 0
        if new_paths:
            models = self._loaded_models()
            checkpoints = run_pipeline(
                paths=new_paths,
                folder_lookup=self._find_folder,
                models=models,
                progress_callback=progress_callback,
                metadata_lookup=metadata_lookup or None,
            )

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
                self._save_state()

        self._last_scan = time.time()
        self._save_state()

        return {
            "new": total_added - len(changed_indices),
            "changed": len(changed_indices),
            "deleted": len(deleted_indices),
            "total": len(self._state.images),
        }

    def _find_folder(self, path: Path) -> str:
        resolved = str(path.resolve())
        if self._state.photos_library and resolved.startswith(self._state.photos_library):
            return PHOTOS_LIBRARY_SOURCE
        for folder in self._state.folders:
            if resolved.startswith(folder):
                return folder
        return str(path.parent)

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

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        # Snapshot mutable state so concurrent refresh can't cause IndexError.
        images = list(self._state.images)
        embeddings = {k: v for k, v in self._embeddings.items()}
        meta_embeddings = {k: v for k, v in self._meta_embeddings.items()}
        annotations = dict(self._annotations)

        if not images:
            return []

        text_models = self._text_capable_models()
        if not text_models:
            return []

        n = len(images)
        rankings: dict[str, list[tuple[int, float]]] = {}
        raw_scores: dict[str, np.ndarray] = {}

        for model_name in text_models:
            model = self._ensure_model(model_name)
            query_vec = model.encode_text(query)

            # Visual similarity
            emb = embeddings.get(model_name)
            if emb is not None and len(emb) >= n:
                emb = emb[:n]
                scores = emb @ query_vec
                raw_scores[model_name] = scores
                ranked_indices = np.argsort(scores)[::-1]
                rankings[model_name] = [(int(i), float(scores[i])) for i in ranked_indices]

            # Metadata text similarity
            meta_emb = meta_embeddings.get(model_name)
            if meta_emb is not None and len(meta_emb) >= n:
                meta_emb = meta_emb[:n]
                meta_scores = meta_emb @ query_vec
                meta_key = f"{model_name}{META_SUFFIX}"
                raw_scores[meta_key] = meta_scores
                ranked_meta = np.argsort(meta_scores)[::-1]
                rankings[meta_key] = [(int(i), float(meta_scores[i])) for i in ranked_meta]

        # Annotation keyword ranking
        ann_ranking = self._rank_by_annotations(query, images, annotations)
        if ann_ranking:
            rankings["annotations"] = ann_ranking

        # Metadata keyword ranking (people names, keywords, filenames, etc.)
        meta_kw_ranking = self._rank_by_metadata_keywords(query, images)
        if meta_kw_ranking:
            rankings["metadata_keywords"] = meta_kw_ranking

        if not rankings:
            return []

        weights = {m: 1.0 for m in rankings}
        if "annotations" in weights:
            weights["annotations"] = ANNOTATION_WEIGHT
        if "metadata_keywords" in weights:
            weights["metadata_keywords"] = ANNOTATION_WEIGHT

        fused = weighted_rrf(rankings, weights, n, top_k)

        results = []
        for idx, fused_score in fused:
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
            ))
        return results

    # -- full reindex --

    def full_reindex(self, progress_callback: ProgressCallback | None = None) -> dict:
        start = time.time()
        self._state.images.clear()
        self._embeddings.clear()
        self._meta_embeddings.clear()
        stats = self.refresh(progress_callback=progress_callback)
        elapsed = time.time() - start
        return {
            "images": stats["total"],
            "models": self._state.enabled_models,
            "elapsed": elapsed,
        }

    # -- status --

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
            "last_scan": self._last_scan,
        }
