import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb
from openai import OpenAI

from chunker import Chunk, chunk_file
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EXCLUDE_DIRS,
    INCLUDE_EXTENSIONS,
    KEYWORD_WEIGHT,
    RRF_K,
)

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 100


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
    def __init__(self, project_root: Path, chroma_dir: Path | None = None):
        self._project_root = project_root.resolve()
        self._chroma_dir = chroma_dir or CHROMA_DIR
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._openai = OpenAI()
        self._last_scan: float | None = None

        # FTS5 keyword index alongside ChromaDB
        fts_path = self._chroma_dir / "fts.db"
        self._fts_conn = sqlite3.connect(str(fts_path))
        self._fts_conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(id, content)"
        )
        # Backfill FTS from ChromaDB if FTS is empty but ChromaDB has data
        fts_count = self._fts_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if fts_count == 0 and self._collection.count() > 0:
            logger.info("Backfilling FTS index from ChromaDB (%d chunks)", self._collection.count())
            all_data = self._collection.get(include=["documents"])
            self._fts_upsert(all_data["ids"], all_data["documents"])

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
        """Return {rel_path: mtime} for all indexed files."""
        if self._collection.count() == 0:
            return {}

        all_meta = self._collection.get(include=["metadatas"])
        file_mtimes: dict[str, float] = {}
        for meta in all_meta["metadatas"]:
            rp = meta["rel_path"]
            mt = meta["mtime"]
            if rp not in file_mtimes or mt > file_mtimes[rp]:
                file_mtimes[rp] = mt
        return file_mtimes

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
        """Embed texts via OpenAI API in batches."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            response = self._openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            all_embeddings.extend(e.embedding for e in response.data)
        return all_embeddings

    # -- FTS5 helpers --

    def _fts_upsert(self, ids: list[str], texts: list[str]) -> None:
        self._fts_conn.executemany(
            "DELETE FROM chunks WHERE id = ?", [(id_,) for id_ in ids]
        )
        self._fts_conn.executemany(
            "INSERT INTO chunks (id, content) VALUES (?, ?)",
            zip(ids, texts),
        )
        self._fts_conn.commit()

    def _fts_delete_by_prefix(self, prefix: str) -> None:
        self._fts_conn.execute(
            "DELETE FROM chunks WHERE id >= ? AND id < ?",
            (prefix, prefix + "\uffff"),
        )
        self._fts_conn.commit()

    def _fts_drop_all(self) -> None:
        self._fts_conn.execute("DELETE FROM chunks")
        self._fts_conn.commit()

    def _fts_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Search FTS5 index, return [(chunk_id, bm25_rank), ...]."""
        safe = _fts_escape(query)
        if not safe:
            return []
        return self._fts_conn.execute(
            "SELECT id, rank FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
            (safe, limit),
        ).fetchall()

    def index_files(self, paths: list[Path]) -> int:
        """Chunk, embed, and upsert files. Returns chunk count."""
        if not paths:
            return 0

        all_chunks: list[Chunk] = []
        for p in paths:
            all_chunks.extend(chunk_file(p, self._project_root))

        if not all_chunks:
            return 0

        texts = [c.text for c in all_chunks]
        embeddings = self._embed_texts(texts)

        ids = [f"{c.rel_path}::{c.chunk_index}" for c in all_chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "rel_path": c.rel_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "file_type": c.file_type,
                "mtime": os.path.getmtime(c.file_path),
                "chunk_index": c.chunk_index,
            }
            for c in all_chunks
        ]

        # ChromaDB upsert in batches (max 5461 per call)
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        self._fts_upsert(ids, texts)
        return len(all_chunks)

    def remove_deleted(self, rel_paths: list[str]) -> int:
        """Remove all chunks for files that no longer exist."""
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
            self._fts_delete_by_prefix(rp + "::")
        return removed

    def refresh(self) -> dict:
        """Mtime-based refresh. Returns stats."""
        to_reindex, to_delete = self.get_stale_files()

        # Remove old chunks for files being reindexed
        reindex_rels = [
            str(p.resolve().relative_to(self._project_root)) for p in to_reindex
        ]
        for rp in reindex_rels:
            existing = self._collection.get(where={"rel_path": rp}, include=[])
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
            self._fts_delete_by_prefix(rp + "::")

        removed = self.remove_deleted(to_delete)
        indexed = self.index_files(to_reindex)

        self._last_scan = time.time()

        total_files = len(set(
            m["rel_path"] for m in self._collection.get(include=["metadatas"])["metadatas"]
        )) if self._collection.count() > 0 else 0

        return {
            "reindexed_files": len(to_reindex),
            "reindexed_chunks": indexed,
            "removed_files": len(to_delete),
            "removed_chunks": removed,
            "total_files": total_files,
            "total_chunks": self._collection.count(),
        }

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """Hybrid search: vector similarity fused with BM25 keyword matching."""
        if self._collection.count() == 0:
            return []

        fetch_k = min(top_k * 3, self._collection.count())

        # Vector ranking
        query_embedding = self._embed_texts([query])[0]
        vector_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )
        vector_ids = vector_results["ids"][0]

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

        # Build lookup from vector results
        id_to_data: dict[str, tuple[str, dict]] = {}
        for cid, doc, meta in zip(
            vector_ids,
            vector_results["documents"][0],
            vector_results["metadatas"][0],
        ):
            id_to_data[cid] = (doc, meta)

        # Fetch any keyword-only hits not in vector results
        missing_ids = [cid for cid, _ in fused if cid not in id_to_data]
        if missing_ids:
            fetched = self._collection.get(
                ids=missing_ids, include=["documents", "metadatas"]
            )
            for cid, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            ):
                id_to_data[cid] = (doc, meta)

        # Assemble results in fused order
        search_results = []
        for chunk_id, fused_score in fused:
            if chunk_id not in id_to_data:
                continue
            doc, meta = id_to_data[chunk_id]
            search_results.append(
                SearchResult(
                    text=doc,
                    rel_path=meta["rel_path"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    score=fused_score,
                )
            )
        return search_results

    def full_reindex(self) -> dict:
        """Drop and rebuild."""
        start = time.time()

        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._fts_drop_all()

        files = self.scan_files()
        chunks = self.index_files(files)
        elapsed = time.time() - start
        self._last_scan = time.time()

        return {"files": len(files), "chunks": chunks, "elapsed": elapsed}

    def status(self) -> dict:
        """Index statistics."""
        total_chunks = self._collection.count()

        if total_chunks > 0:
            all_meta = self._collection.get(include=["metadatas"])
            rel_paths = set(m["rel_path"] for m in all_meta["metadatas"])
            total_files = len(rel_paths)
        else:
            total_files = 0

        to_reindex, to_delete = self.get_stale_files()

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "last_scan": self._last_scan,
            "pending_reindex": len(to_reindex) + len(to_delete),
        }
