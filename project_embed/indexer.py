import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from chunker import Chunk, chunk_file
from config import (
    STORE_DIR,
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EXCLUDE_DIRS,
    INCLUDE_EXTENSIONS,
    KEYWORD_WEIGHT,
    RRF_K,
)

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 100
HNSW_EF_CONSTRUCTION = 200
HNSW_M = 16
HNSW_EF_SEARCH = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    rel_path    TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    file_type   TEXT NOT NULL,
    mtime       REAL NOT NULL,
    chunk_index INTEGER NOT NULL,
    hnsw_label  INTEGER NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path);
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_label ON chunks(hnsw_label);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id, content);
"""


def rrf(
    rankings: dict[str, list[str]],
    weights: dict[str, float],
    top_k: int,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of chunk IDs."""
    scores: dict[str, float] = {}
    for source, ranked_ids in rankings.items():
        w = weights.get(source, 1.0)
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w / (RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _fts_escape(query: str) -> str:
    """Convert natural language query to safe FTS5 query tokens."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", query)
    return " ".join(tokens)


@dataclass
class SearchResult:
    text: str
    rel_path: str
    start_line: int
    end_line: int
    score: float


class CodeIndex:
    def __init__(self, project_root: Path, store_dir: Path | None = None):
        self._project_root = project_root.resolve()
        self._store_dir = store_dir or STORE_DIR
        self._store_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._store_dir / "store.db"
        self._hnsw_path = self._store_dir / "index.bin"

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.executescript(_SCHEMA)

        self._embedding_provider = EMBEDDING_PROVIDER
        self._openai = OpenAI() if self._embedding_provider == "openai" else None
        self._local_model = (
            SentenceTransformer("all-MiniLM-L6-v2")
            if self._embedding_provider == "local"
            else None
        )

        self._hnsw = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
        self._next_label = self._load_next_label()
        self._load_hnsw()

        self._last_scan: float | None = None

    def _load_next_label(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'next_label'"
        ).fetchone()
        if row:
            return int(row[0])
        max_label = self._conn.execute(
            "SELECT MAX(hnsw_label) FROM chunks"
        ).fetchone()[0]
        return (max_label + 1) if max_label is not None else 0

    def _save_next_label(self) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('next_label', ?)",
            (str(self._next_label),),
        )
        self._conn.commit()

    def _load_hnsw(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if count > 0 and self._hnsw_path.exists():
            self._hnsw.load_index(str(self._hnsw_path), max_elements=count + 10000)
            self._hnsw.set_ef(HNSW_EF_SEARCH)
        else:
            self._hnsw.init_index(
                max_elements=max(count + 10000, 1000),
                ef_construction=HNSW_EF_CONSTRUCTION,
                M=HNSW_M,
            )
            self._hnsw.set_ef(HNSW_EF_SEARCH)

    def _save_hnsw(self) -> None:
        self._hnsw.save_index(str(self._hnsw_path))

    def _allocate_labels(self, n: int) -> list[int]:
        labels = list(range(self._next_label, self._next_label + n))
        self._next_label += n
        self._save_next_label()
        return labels

    def _chunk_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def scan_files(self) -> list[Path]:
        """Walk project tree, collect indexable files."""
        results = []
        for dirpath, dirnames, filenames in os.walk(self._project_root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fname in filenames:
                p = Path(dirpath) / fname
                if p.suffix.lower() in INCLUDE_EXTENSIONS:
                    results.append(p)
        return results

    def _indexed_files(self) -> dict[str, float]:
        """Return {rel_path: max_mtime} for all indexed files."""
        rows = self._conn.execute(
            "SELECT rel_path, MAX(mtime) FROM chunks GROUP BY rel_path"
        ).fetchall()
        return {rp: mt for rp, mt in rows}

    def get_stale_files(self) -> tuple[list[Path], list[str]]:
        """Compare mtimes. Returns (files_to_reindex, rel_paths_to_delete)."""
        current_files = self.scan_files()
        indexed = self._indexed_files()

        current_by_rel: dict[str, Path] = {}
        for p in current_files:
            rel = str(p.resolve().relative_to(self._project_root))
            current_by_rel[rel] = p

        to_reindex: list[Path] = []
        for rel, path in current_by_rel.items():
            stored_mtime = indexed.get(rel)
            if stored_mtime is None or os.path.getmtime(path) > stored_mtime:
                to_reindex.append(path)

        to_delete = [rp for rp in indexed if rp not in current_by_rel]

        return to_reindex, to_delete

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via local model or OpenAI API in batches."""
        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        for batch_index, i in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), start=1):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            logger.info(
                "Embedding batch %d/%d (%d texts) via %s",
                batch_index,
                total_batches,
                len(batch),
                self._embedding_provider,
            )
            if self._embedding_provider == "local":
                assert self._local_model is not None
                vecs = self._local_model.encode(batch, normalize_embeddings=True)
                all_embeddings.extend(vecs.tolist())
            else:
                assert self._openai is not None
                response = self._openai.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                all_embeddings.extend(e.embedding for e in response.data)
        return all_embeddings

    # -- FTS5 helpers --

    def _fts_upsert(self, ids: list[str], texts: list[str]) -> None:
        self._conn.executemany(
            "DELETE FROM chunks_fts WHERE id = ?", [(id_,) for id_ in ids]
        )
        self._conn.executemany(
            "INSERT INTO chunks_fts (id, content) VALUES (?, ?)",
            zip(ids, texts),
        )

    def _fts_delete_by_prefix(self, prefix: str) -> None:
        self._conn.execute(
            "DELETE FROM chunks_fts WHERE id >= ? AND id < ?",
            (prefix, prefix + "\uffff"),
        )

    def _fts_drop_all(self) -> None:
        self._conn.execute("DELETE FROM chunks_fts")

    def _fts_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Search FTS5 index, return [(chunk_id, bm25_rank), ...]."""
        safe = _fts_escape(query)
        if not safe:
            return []
        return self._conn.execute(
            "SELECT id, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (safe, limit),
        ).fetchall()

    def index_files(self, paths: list[Path]) -> int:
        """Chunk, embed, and upsert files. Returns chunk count."""
        if not paths:
            return 0

        logger.info("Chunking %d files", len(paths))
        all_chunks: list[Chunk] = []
        for p in paths:
            all_chunks.extend(chunk_file(p, self._project_root))

        if not all_chunks:
            return 0

        logger.info("Prepared %d chunks for embedding", len(all_chunks))
        texts = [c.text for c in all_chunks]
        embeddings = self._embed_texts(texts)

        ids = [f"{c.rel_path}::{c.chunk_index}" for c in all_chunks]
        labels = self._allocate_labels(len(all_chunks))

        # Resize HNSW if needed
        current_max = self._hnsw.get_max_elements()
        needed = self._hnsw.get_current_count() + len(all_chunks)
        if needed > current_max:
            self._hnsw.resize_index(needed + 10000)

        # Add to HNSW index
        vectors = np.array(embeddings, dtype=np.float32)
        self._hnsw.add_items(vectors, labels)

        # Upsert into SQLite
        rows = [
            (
                ids[i],
                texts[i],
                all_chunks[i].rel_path,
                all_chunks[i].file_path,
                all_chunks[i].start_line,
                all_chunks[i].end_line,
                all_chunks[i].file_type,
                os.path.getmtime(all_chunks[i].file_path),
                all_chunks[i].chunk_index,
                labels[i],
            )
            for i in range(len(all_chunks))
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (id, content, rel_path, file_path, start_line, end_line,
                file_type, mtime, chunk_index, hnsw_label)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

        # Update FTS
        self._fts_upsert(ids, texts)
        self._conn.commit()
        self._save_hnsw()

        return len(all_chunks)

    def _delete_chunks_for_rel_path(self, rel_path: str) -> int:
        """Delete all chunks for a given rel_path. Returns count removed."""
        rows = self._conn.execute(
            "SELECT id FROM chunks WHERE rel_path = ?", (rel_path,)
        ).fetchall()
        if not rows:
            return 0
        chunk_ids = [r[0] for r in rows]
        self._conn.execute("DELETE FROM chunks WHERE rel_path = ?", (rel_path,))
        self._fts_delete_by_prefix(rel_path + "::")
        return len(chunk_ids)

    def remove_deleted(self, rel_paths: list[str]) -> int:
        """Remove all chunks for files that no longer exist."""
        if not rel_paths:
            return 0
        removed = 0
        for rp in rel_paths:
            removed += self._delete_chunks_for_rel_path(rp)
        self._conn.commit()
        return removed

    def refresh(self) -> dict:
        """Mtime-based refresh. Returns stats."""
        to_reindex, to_delete = self.get_stale_files()

        # Remove old chunks for files being reindexed
        for p in to_reindex:
            rp = str(p.resolve().relative_to(self._project_root))
            self._delete_chunks_for_rel_path(rp)
        self._conn.commit()

        removed = self.remove_deleted(to_delete)
        indexed = self.index_files(to_reindex)

        self._last_scan = time.time()

        total_files = self._conn.execute(
            "SELECT COUNT(DISTINCT rel_path) FROM chunks"
        ).fetchone()[0]

        return {
            "reindexed_files": len(to_reindex),
            "reindexed_chunks": indexed,
            "removed_files": len(to_delete),
            "removed_chunks": removed,
            "total_files": total_files,
            "total_chunks": self._chunk_count(),
        }

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """Hybrid search: vector similarity fused with BM25 keyword matching."""
        count = self._chunk_count()
        if count == 0:
            return []

        fetch_k = min(top_k * 3, count)

        # Vector ranking
        query_embedding = self._embed_texts([query])[0]
        query_vec = np.array([query_embedding], dtype=np.float32)
        labels_arr, distances_arr = self._hnsw.knn_query(query_vec, k=fetch_k)

        # Map HNSW labels back to chunk IDs
        labels_list = labels_arr[0].tolist()
        placeholders = ",".join("?" * len(labels_list))
        rows = self._conn.execute(
            f"SELECT hnsw_label, id, content, rel_path, start_line, end_line FROM chunks WHERE hnsw_label IN ({placeholders})",
            labels_list,
        ).fetchall()
        label_to_data = {r[0]: r[1:] for r in rows}

        vector_ids = []
        id_to_data: dict[str, tuple[str, str, int, int]] = {}
        for label in labels_list:
            data = label_to_data.get(label)
            if data:
                chunk_id, content, rel_path, start_line, end_line = data
                vector_ids.append(chunk_id)
                id_to_data[chunk_id] = (content, rel_path, start_line, end_line)

        # BM25 keyword ranking
        fts_hits = self._fts_search(query, limit=fetch_k)
        keyword_ids = [chunk_id for chunk_id, _rank in fts_hits]

        # Fuse rankings
        rankings: dict[str, list[str]] = {"vector": vector_ids}
        weights = {"vector": 1.0}
        if keyword_ids:
            rankings["keyword"] = keyword_ids
            weights["keyword"] = KEYWORD_WEIGHT

        fused = rrf(rankings, weights, top_k)

        # Fetch any keyword-only hits not in vector results
        missing_ids = [cid for cid, _ in fused if cid not in id_to_data]
        if missing_ids:
            placeholders = ",".join("?" * len(missing_ids))
            rows = self._conn.execute(
                f"SELECT id, content, rel_path, start_line, end_line FROM chunks WHERE id IN ({placeholders})",
                missing_ids,
            ).fetchall()
            for chunk_id, content, rel_path, start_line, end_line in rows:
                id_to_data[chunk_id] = (content, rel_path, start_line, end_line)

        # Assemble results in fused order
        search_results = []
        for chunk_id, fused_score in fused:
            if chunk_id not in id_to_data:
                continue
            content, rel_path, start_line, end_line = id_to_data[chunk_id]
            search_results.append(
                SearchResult(
                    text=content,
                    rel_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    score=fused_score,
                )
            )
        return search_results

    def full_reindex(self) -> dict:
        """Drop and rebuild."""
        start = time.time()

        logger.info("Dropping existing index")
        self._conn.execute("DELETE FROM chunks")
        self._fts_drop_all()
        self._conn.execute("DELETE FROM meta")
        self._conn.commit()

        # Reset HNSW
        if self._hnsw_path.exists():
            self._hnsw_path.unlink()
        self._hnsw = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
        self._hnsw.init_index(
            max_elements=1000,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
        )
        self._hnsw.set_ef(HNSW_EF_SEARCH)
        self._next_label = 0
        self._save_next_label()

        files = self.scan_files()
        logger.info("Scanned %d files under %s", len(files), self._project_root)
        chunks = self.index_files(files)
        elapsed = time.time() - start
        self._last_scan = time.time()

        return {"files": len(files), "chunks": chunks, "elapsed": elapsed}

    def status(self) -> dict:
        """Index statistics."""
        total_chunks = self._chunk_count()
        total_files = self._conn.execute(
            "SELECT COUNT(DISTINCT rel_path) FROM chunks"
        ).fetchone()[0]

        to_reindex, to_delete = self.get_stale_files()

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "last_scan": self._last_scan,
            "pending_reindex": len(to_reindex) + len(to_delete),
        }
