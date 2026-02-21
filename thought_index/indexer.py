import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb

from chunker import Chunk, chunk_file
from embedder import GpuEmbeddingFunction
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    DIVERSITY_OVERFETCH,
    EXCLUDE_DIRS,
    INCLUDE_EXTENSIONS,
    MAX_PER_FILE,
    REFRESH_DEBOUNCE_SECONDS,
)
from sources import Source, load_sources

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    rel_path: str
    start_line: int
    end_line: int
    score: float


class ThoughtIndex:
    _GET_PAGE_SIZE = 500

    def __init__(self, chroma_dir: Path | None = None):
        self._chroma_dir = chroma_dir or CHROMA_DIR
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._embed_fn = GpuEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_fn,
        )
        self._last_scan: float | None = None
        self._refresh_lock = threading.Lock()
        self._refreshing = False

    def _get_all(self, include: list[str], where: dict | None = None) -> dict:
        """Paginated collection.get() to avoid SQLite variable limit."""
        combined: dict[str, list] = {k: [] for k in include}
        combined["ids"] = []
        offset = 0
        while True:
            kwargs: dict = {
                "include": include,
                "limit": self._GET_PAGE_SIZE,
                "offset": offset,
            }
            if where is not None:
                kwargs["where"] = where
            page = self._collection.get(**kwargs)
            if not page["ids"]:
                break
            combined["ids"].extend(page["ids"])
            for key in include:
                combined[key].extend(page[key])
            if len(page["ids"]) < self._GET_PAGE_SIZE:
                break
            offset += self._GET_PAGE_SIZE
        return combined

    def scan_files(self) -> list[tuple[Path, Source]]:
        sources = load_sources()
        results = []
        for source in sources:
            root = Path(source.path)
            if not root.is_dir():
                logger.warning("Source folder missing: %s", source.path)
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
                for fname in filenames:
                    p = Path(dirpath) / fname
                    if p.suffix.lower() in INCLUDE_EXTENSIONS:
                        results.append((p, source))
        return results

    def _rel_path(self, path: Path, source: Source) -> str:
        inner = str(path.resolve().relative_to(Path(source.path).resolve()))
        return f"{source.label}/{inner}"

    def _indexed_files(self) -> dict[str, float]:
        if self._collection.count() == 0:
            return {}

        all_meta = self._get_all(include=["metadatas"])
        file_mtimes: dict[str, float] = {}
        for meta in all_meta["metadatas"]:
            rp = meta["rel_path"]
            mt = meta["mtime"]
            if rp not in file_mtimes or mt > file_mtimes[rp]:
                file_mtimes[rp] = mt
        return file_mtimes

    def get_stale_files(self) -> tuple[list[tuple[Path, Source]], list[str]]:
        current_files = self.scan_files()
        indexed = self._indexed_files()

        current_by_rel: dict[str, tuple[Path, Source]] = {}
        for path, source in current_files:
            rel = self._rel_path(path, source)
            current_by_rel[rel] = (path, source)

        to_reindex: list[tuple[Path, Source]] = []
        for rel, (path, source) in current_by_rel.items():
            stored_mtime = indexed.get(rel)
            if stored_mtime is None or os.path.getmtime(path) > stored_mtime:
                to_reindex.append((path, source))

        to_delete = [rp for rp in indexed if rp not in current_by_rel]

        return to_reindex, to_delete

    def index_files(
        self,
        file_source_pairs: list[tuple[Path, Source]],
        progress_callback=None,
    ) -> int:
        if not file_source_pairs:
            return 0

        total_files = len(file_source_pairs)
        all_chunks: list[Chunk] = []
        for i, (path, source) in enumerate(file_source_pairs):
            all_chunks.extend(
                chunk_file(path, Path(source.path), source.label)
            )
            if progress_callback:
                progress_callback(i + 1, total_files, path.name)

        if not all_chunks:
            return 0

        texts = [c.text for c in all_chunks]
        ids = [f"{c.rel_path}::{c.chunk_index}" for c in all_chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "rel_path": c.rel_path,
                "source_label": c.rel_path.split("/", 1)[0],
                "start_line": c.start_line,
                "end_line": c.end_line,
                "file_type": c.file_type,
                "mtime": os.path.getmtime(c.file_path),
                "chunk_index": c.chunk_index,
            }
            for c in all_chunks
        ]

        if progress_callback:
            progress_callback(total_files, total_files, f"Upserting {len(all_chunks)} chunks")

        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        return len(all_chunks)

    def remove_deleted(self, rel_paths: list[str]) -> int:
        if not rel_paths:
            return 0

        removed = 0
        for rp in rel_paths:
            existing = self._collection.get(
                where={"rel_path": rp}, include=[]
            )
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                removed += len(existing["ids"])
        return removed

    def remove_source_chunks(self, source_label: str) -> int:
        existing = self._get_all(
            include=[], where={"source_label": source_label}
        )
        if existing["ids"]:
            batch_size = self._GET_PAGE_SIZE
            for i in range(0, len(existing["ids"]), batch_size):
                self._collection.delete(ids=existing["ids"][i:i + batch_size])
        return len(existing["ids"])

    def index_source(self, source: Source, progress_callback=None) -> dict:
        root = Path(source.path)
        if not root.is_dir():
            return {"files": 0, "chunks": 0}

        pairs: list[tuple[Path, Source]] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fname in filenames:
                p = Path(dirpath) / fname
                if p.suffix.lower() in INCLUDE_EXTENSIONS:
                    pairs.append((p, source))

        chunks = self.index_files(pairs, progress_callback=progress_callback)
        return {"files": len(pairs), "chunks": chunks}

    def source_file_count(self, source_label: str) -> int:
        if self._collection.count() == 0:
            return 0
        existing = self._get_all(
            include=["metadatas"], where={"source_label": source_label}
        )
        rel_paths = set(m["rel_path"] for m in existing["metadatas"])
        return len(rel_paths)

    def _needs_refresh(self) -> bool:
        if self._last_scan is None:
            return True
        return (time.time() - self._last_scan) >= REFRESH_DEBOUNCE_SECONDS

    def _do_refresh(self, progress_callback=None) -> None:
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            self._refreshing = True
            to_reindex, to_delete = self.get_stale_files()

            reindex_rels = [
                self._rel_path(p, s) for p, s in to_reindex
            ]
            for rp in reindex_rels:
                existing = self._collection.get(where={"rel_path": rp}, include=[])
                if existing["ids"]:
                    self._collection.delete(ids=existing["ids"])

            removed = self.remove_deleted(to_delete)
            indexed = self.index_files(to_reindex, progress_callback=progress_callback)

            self._last_scan = time.time()

            if to_reindex or to_delete:
                logger.info(
                    "Background refresh: %d files re-indexed (%d chunks), %d deleted (%d chunks)",
                    len(to_reindex), indexed, len(to_delete), removed,
                )
        except Exception:
            logger.exception("Background refresh failed")
        finally:
            self._refreshing = False
            self._refresh_lock.release()

    def schedule_refresh(self, progress_callback=None) -> None:
        """Trigger a background refresh if the index is stale."""
        if not self._needs_refresh():
            return
        if self._refreshing:
            return
        thread = threading.Thread(
            target=self._do_refresh, args=(progress_callback,), daemon=True,
        )
        thread.start()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        if self._collection.count() == 0:
            return []

        fetch_n = min(top_k * DIVERSITY_OVERFETCH, self._collection.count())
        results = self._collection.query(
            query_texts=[query],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )

        file_counts: dict[str, int] = {}
        search_results = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            rel_path = meta["rel_path"]
            if file_counts.get(rel_path, 0) >= MAX_PER_FILE:
                continue
            file_counts[rel_path] = file_counts.get(rel_path, 0) + 1
            search_results.append(
                SearchResult(
                    text=doc,
                    rel_path=rel_path,
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    score=1.0 - dist,
                )
            )
            if len(search_results) >= top_k:
                break
        return search_results

    def full_reindex(self, progress_callback=None) -> dict:
        start = time.time()

        # Clear all documents instead of dropping collection
        # (drop triggers SQLite operations that fail in sandboxed environments)
        all_ids = self._get_all(include=[])["ids"]
        batch_size = self._GET_PAGE_SIZE
        for i in range(0, len(all_ids), batch_size):
            self._collection.delete(ids=all_ids[i : i + batch_size])

        file_source_pairs = self.scan_files()
        chunks = self.index_files(file_source_pairs, progress_callback=progress_callback)
        elapsed = time.time() - start
        self._last_scan = time.time()

        sources = load_sources()
        return {
            "files": len(file_source_pairs),
            "chunks": chunks,
            "sources": len(sources),
            "elapsed": elapsed,
        }

    def status(self) -> dict:
        total_chunks = self._collection.count()

        if total_chunks > 0:
            all_meta = self._get_all(include=["metadatas"])
            rel_paths = set(m["rel_path"] for m in all_meta["metadatas"])
            total_files = len(rel_paths)
        else:
            total_files = 0

        to_reindex, to_delete = self.get_stale_files()
        sources = load_sources()

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "last_scan": self._last_scan,
            "source_count": len(sources),
            "pending_reindex": len(to_reindex) + len(to_delete),
        }
