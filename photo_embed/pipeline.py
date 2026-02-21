"""Streaming indexing pipeline.

Overlaps CPU-bound thumbnail generation with GPU-bound embedding computation
using a producer/consumer pattern with threads.

    ThreadPool (CPU)          Main Thread (GPU)
    ================          =================
    thumbnail(img_1)
    + extract metadata
    thumbnail(img_2)    -->   embed_batch([1..16], model_a)  (images)
    thumbnail(img_3)    -->   embed_batch([1..16], model_a)  (metadata text)
    ...                       checkpoint every SAVE_INTERVAL
"""

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

import numpy as np
from PIL import Image

from config import (
    BATCH_PAUSE_SECONDS,
    EMBEDDING_BATCH_SIZE,
    SAVE_INTERVAL,
)
from metadata import extract_metadata
from models import EmbeddingModel
from thumbnails import (
    model_thumbnail,
    open_for_embedding,
    preview_thumbnail,
    register_heif,
    thumb_key,
)

logger = logging.getLogger(__name__)

THUMBNAIL_WORKERS = 4
PREP_QUEUE_MAXSIZE = EMBEDDING_BATCH_SIZE * 4


@dataclass
class ImageRecord:
    path: str
    folder: str
    mtime: float
    size: int
    thumb_key: str
    metadata_text: str = ""
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class PreparedImage:
    record: ImageRecord
    image: Image.Image


@dataclass
class Checkpoint:
    """Yielded after every SAVE_INTERVAL images."""

    records: list[ImageRecord]
    embeddings: dict[str, np.ndarray]  # model_name -> (N, D)
    meta_embeddings: dict[str, np.ndarray]  # model_name -> (N, D)
    total_done: int
    total_target: int
    rate: float  # photos/sec since pipeline start


ProgressCallback = Callable[[int, int, str], None]


def _prepare_one(
    path: Path,
    folder: str,
    extra_metadata: dict | None = None,
) -> PreparedImage | None:
    """Generate thumbnails, extract metadata, open image. Runs in thread pool."""
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
        key = thumb_key(path, mtime)
        model_thumbnail(path, key)
        preview_thumbnail(path, key)
        thumb_path = model_thumbnail(path, key)
        image = open_for_embedding(thumb_path)
        meta_text, lat, lon = extract_metadata(path, extra=extra_metadata)
        record = ImageRecord(
            path=str(path.resolve()),
            folder=folder,
            mtime=mtime,
            size=size,
            thumb_key=key,
            metadata_text=meta_text,
            latitude=lat,
            longitude=lon,
        )
        return PreparedImage(record=record, image=image)
    except Exception:
        logger.warning("Failed to prepare %s", path, exc_info=True)
        return None


def _text_capable(models: dict[str, EmbeddingModel]) -> dict[str, EmbeddingModel]:
    """Filter to models that support text encoding."""
    return {n: m for n, m in models.items() if m.supports_text}


def run_pipeline(
    paths: list[Path],
    folder_lookup: Callable[[Path], str],
    models: dict[str, EmbeddingModel],
    progress_callback: ProgressCallback | None = None,
    metadata_lookup: dict[str, dict] | None = None,
):
    """Run the streaming indexing pipeline.

    Yields Checkpoints as they are produced (every SAVE_INTERVAL images)
    so callers can persist progress incrementally.

    Args:
        paths: Image file paths to process.
        folder_lookup: Maps a Path to its source folder string.
        models: {name: loaded EmbeddingModel} for embedding.
        progress_callback: Optional (current, total, detail) reporter.
        metadata_lookup: Optional {resolved_path_str: metadata_dict} for extra
            metadata from Apple Photos or other sources.

    Yields:
        Checkpoint after every SAVE_INTERVAL images and at the end.
    """
    if not paths:
        return

    register_heif()
    total = len(paths)
    start_time = time.time()
    text_models = _text_capable(models)

    if progress_callback:
        progress_callback(0, total, "Starting")

    # -- Stage 1: submit all thumbnail + metadata work to thread pool --
    pool = ThreadPoolExecutor(max_workers=THUMBNAIL_WORKERS)

    futures: list[Future] = []
    for p in paths:
        folder = folder_lookup(p)
        extra = metadata_lookup.get(str(p.resolve())) if metadata_lookup else None
        f = pool.submit(_prepare_one, p, folder, extra)
        futures.append(f)

    # -- Stage 2: consume prepared images, batch, embed --
    total_done = 0
    checkpoint_records: list[ImageRecord] = []
    checkpoint_embeddings: dict[str, list[np.ndarray]] = {
        name: [] for name in models
    }
    checkpoint_meta_embeddings: dict[str, list[np.ndarray]] = {
        name: [] for name in text_models
    }
    batch_images: list[Image.Image] = []
    batch_records: list[ImageRecord] = []

    def _flush_batch() -> None:
        """Embed current batch through all models (images + metadata text)."""
        if not batch_images:
            return

        # Image embeddings
        for model_name, model in models.items():
            emb = model.encode_images(batch_images)
            checkpoint_embeddings[model_name].append(emb)

        # Metadata text embeddings (text-capable models only)
        if text_models:
            texts = [r.metadata_text for r in batch_records]
            has_text = any(t for t in texts)
            for model_name, model in text_models.items():
                if has_text:
                    # Replace empty strings with a neutral placeholder
                    # so batch size matches; zero-norm will produce zero similarity
                    safe_texts = [t if t else "." for t in texts]
                    meta_emb = model.encode_texts(safe_texts)
                    # Zero out embeddings for images with no metadata
                    for i, t in enumerate(texts):
                        if not t:
                            meta_emb[i] = 0.0
                    checkpoint_meta_embeddings[model_name].append(meta_emb)
                else:
                    dim = model.embedding_dim
                    checkpoint_meta_embeddings[model_name].append(
                        np.zeros((len(batch_images), dim), dtype=np.float32)
                    )

        checkpoint_records.extend(batch_records)
        batch_images.clear()
        batch_records.clear()

    def _make_checkpoint() -> Checkpoint | None:
        nonlocal checkpoint_records
        nonlocal checkpoint_embeddings, checkpoint_meta_embeddings

        if not checkpoint_records:
            return None

        merged: dict[str, np.ndarray] = {}
        for model_name, parts in checkpoint_embeddings.items():
            if parts:
                merged[model_name] = np.vstack(parts)

        merged_meta: dict[str, np.ndarray] = {}
        for model_name, parts in checkpoint_meta_embeddings.items():
            if parts:
                merged_meta[model_name] = np.vstack(parts)

        elapsed = time.time() - start_time
        rate = total_done / elapsed if elapsed > 0 else 0.0

        detail = f"{rate:.1f} photos/sec"
        logger.info(
            "Checkpoint: %d/%d [%s]", total_done, total, detail,
        )
        if progress_callback:
            progress_callback(total_done, total, detail)

        cp = Checkpoint(
            records=list(checkpoint_records),
            embeddings=merged,
            meta_embeddings=merged_meta,
            total_done=total_done,
            total_target=total,
            rate=rate,
        )

        checkpoint_records = []
        checkpoint_embeddings = {name: [] for name in models}
        checkpoint_meta_embeddings = {name: [] for name in text_models}

        return cp

    # Wait for each future and process results as they arrive
    for f in futures:
        try:
            result = f.result()
        except Exception:
            logger.warning("Thumbnail future failed", exc_info=True)
            continue

        if result is None:
            continue

        batch_images.append(result.image)
        batch_records.append(result.record)
        total_done += 1

        if len(batch_images) >= EMBEDDING_BATCH_SIZE:
            _flush_batch()
            time.sleep(BATCH_PAUSE_SECONDS)

        if total_done % SAVE_INTERVAL == 0:
            _flush_batch()
            cp = _make_checkpoint()
            if cp:
                yield cp

    # Flush remaining
    _flush_batch()
    cp = _make_checkpoint()
    if cp:
        yield cp

    pool.shutdown(wait=False)
