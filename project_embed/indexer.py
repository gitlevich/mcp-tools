import logging
import os
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
)

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 100


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
        """Embed query and search ChromaDB."""
        if self._collection.count() == 0:
            return []

        query_embedding = self._embed_texts([query])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            search_results.append(
                SearchResult(
                    text=doc,
                    rel_path=meta["rel_path"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    score=1.0 - dist,  # cosine distance -> similarity
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
