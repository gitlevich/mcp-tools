"""Entity store backed by SQLite for NER results and resolution."""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ner import ExtractedEntity, extract_entities

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('person', 'place', 'org'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_name_type
    ON entity(canonical_name, type);

CREATE TABLE IF NOT EXISTS alias (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    name TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_alias ON alias(entity_id, name);

CREATE TABLE IF NOT EXISTS mention (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    context TEXT,
    span_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_mention_entity ON mention(entity_id);
CREATE INDEX IF NOT EXISTS idx_mention_path ON mention(rel_path);

CREATE TABLE IF NOT EXISTS extracted_chunk (
    rel_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    mtime REAL NOT NULL,
    PRIMARY KEY (rel_path, chunk_index)
);
"""


@dataclass
class Entity:
    id: int
    canonical_name: str
    type: str
    mention_count: int = 0
    aliases: list[str] | None = None


@dataclass
class Mention:
    id: int
    entity_id: int
    source_path: str
    rel_path: str
    start_line: int | None
    end_line: int | None
    context: str | None
    span_text: str | None


def _context_snippet(text: str, char_start: int, char_end: int, radius: int = 80) -> str:
    """Extract a context window around the entity span."""
    start = max(0, char_start - radius)
    end = min(len(text), char_end + radius)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


class EntityStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    # -- Entity CRUD --

    def _find_or_create_entity(self, name: str, entity_type: str) -> int:
        """Find existing entity by exact name+type or create a new one."""
        row = self._conn.execute(
            "SELECT id FROM entity WHERE canonical_name = ? AND type = ?",
            (name, entity_type),
        ).fetchone()
        if row:
            return row[0]

        # Check aliases
        row = self._conn.execute(
            """SELECT a.entity_id FROM alias a
               JOIN entity e ON e.id = a.entity_id
               WHERE a.name = ? AND e.type = ?""",
            (name, entity_type),
        ).fetchone()
        if row:
            return row[0]

        cursor = self._conn.execute(
            "INSERT INTO entity (canonical_name, type) VALUES (?, ?)",
            (name, entity_type),
        )
        return cursor.lastrowid

    def _add_mention(
        self,
        entity_id: int,
        source_path: str,
        rel_path: str,
        start_line: int | None,
        end_line: int | None,
        context: str | None,
        span_text: str | None,
    ) -> int:
        cursor = self._conn.execute(
            """INSERT INTO mention
               (entity_id, source_path, rel_path, start_line, end_line, context, span_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entity_id, source_path, rel_path, start_line, end_line, context, span_text),
        )
        return cursor.lastrowid

    # -- Extraction --

    def extract_from_chunks(
        self,
        documents: list[str],
        metadatas: list[dict],
        progress_callback=None,
    ) -> dict:
        """Run NER on chunks and populate entities + mentions.

        Skips chunks already processed (by rel_path + chunk_index + mtime).
        Returns stats dict.
        """
        total = len(documents)
        new_entities = 0
        new_mentions = 0
        skipped = 0

        for i, (text, meta) in enumerate(zip(documents, metadatas)):
            rel_path = meta["rel_path"]
            chunk_index = meta.get("chunk_index", 0)
            mtime = meta.get("mtime", 0.0)

            # Check if already extracted at this mtime
            row = self._conn.execute(
                "SELECT mtime FROM extracted_chunk WHERE rel_path = ? AND chunk_index = ?",
                (rel_path, chunk_index),
            ).fetchone()
            if row and row[0] >= mtime:
                skipped += 1
                if progress_callback:
                    progress_callback(i + 1, total, f"skipped {rel_path}")
                continue

            # Remove old mentions for this chunk if re-extracting
            if row:
                self._remove_chunk_mentions(rel_path, meta.get("start_line"), meta.get("end_line"))

            entities = extract_entities(text)
            source_path = meta.get("file_path", "")

            for ent in entities:
                entity_id = self._find_or_create_entity(ent.name, ent.type)
                context = _context_snippet(text, ent.char_start, ent.char_end)
                self._add_mention(
                    entity_id=entity_id,
                    source_path=source_path,
                    rel_path=rel_path,
                    start_line=meta.get("start_line"),
                    end_line=meta.get("end_line"),
                    context=context,
                    span_text=ent.name,
                )
                new_mentions += 1

            new_entities += len(entities)

            # Mark chunk as extracted
            self._conn.execute(
                """INSERT OR REPLACE INTO extracted_chunk (rel_path, chunk_index, mtime)
                   VALUES (?, ?, ?)""",
                (rel_path, chunk_index, mtime),
            )

            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, total, f"{new_mentions} mentions found")

        self._conn.commit()

        if progress_callback:
            progress_callback(total, total, f"Done: {new_mentions} mentions")

        return {
            "chunks_processed": total - skipped,
            "chunks_skipped": skipped,
            "mentions_added": new_mentions,
        }

    def _remove_chunk_mentions(self, rel_path: str, start_line: int | None, end_line: int | None):
        """Remove mentions from a specific chunk (identified by rel_path + line range)."""
        if start_line is not None and end_line is not None:
            self._conn.execute(
                "DELETE FROM mention WHERE rel_path = ? AND start_line = ? AND end_line = ?",
                (rel_path, start_line, end_line),
            )
        else:
            self._conn.execute(
                "DELETE FROM mention WHERE rel_path = ?",
                (rel_path,),
            )

    # -- Search & Browse --

    def search(self, query: str, entity_type: str | None = None, limit: int = 20) -> list[Entity]:
        """Search entities by name substring (canonical name or alias)."""
        pattern = f"%{query}%"
        params: list = [pattern, pattern]
        if entity_type:
            params.append(entity_type)

        sql = f"""
            SELECT DISTINCT e.id, e.canonical_name, e.type,
                   (SELECT COUNT(*) FROM mention m WHERE m.entity_id = e.id) as cnt
            FROM entity e
            LEFT JOIN alias a ON a.entity_id = e.id
            WHERE (e.canonical_name LIKE ? OR a.name LIKE ?)
            {"AND e.type = ?" if entity_type else ""}
            ORDER BY cnt DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [Entity(id=r[0], canonical_name=r[1], type=r[2], mention_count=r[3]) for r in rows]

    def get(self, entity_id: int) -> Entity | None:
        """Get entity with aliases and mention count."""
        row = self._conn.execute(
            """SELECT e.id, e.canonical_name, e.type,
                      (SELECT COUNT(*) FROM mention m WHERE m.entity_id = e.id)
               FROM entity e WHERE e.id = ?""",
            (entity_id,),
        ).fetchone()
        if not row:
            return None

        aliases = [
            r[0]
            for r in self._conn.execute(
                "SELECT name FROM alias WHERE entity_id = ?", (entity_id,)
            ).fetchall()
        ]
        return Entity(
            id=row[0],
            canonical_name=row[1],
            type=row[2],
            mention_count=row[3],
            aliases=aliases,
        )

    def mentions(self, entity_id: int, limit: int = 50) -> list[Mention]:
        """Get all mentions of an entity."""
        rows = self._conn.execute(
            """SELECT id, entity_id, source_path, rel_path,
                      start_line, end_line, context, span_text
               FROM mention WHERE entity_id = ?
               ORDER BY rel_path, start_line
               LIMIT ?""",
            (entity_id, limit),
        ).fetchall()
        return [
            Mention(
                id=r[0], entity_id=r[1], source_path=r[2], rel_path=r[3],
                start_line=r[4], end_line=r[5], context=r[6], span_text=r[7],
            )
            for r in rows
        ]

    def list_entities(
        self, entity_type: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[Entity]:
        """Browse entities ordered by mention count descending."""
        type_clause = "WHERE e.type = ?" if entity_type else ""
        params: list = []
        if entity_type:
            params.append(entity_type)
        params.extend([limit, offset])

        rows = self._conn.execute(
            f"""SELECT e.id, e.canonical_name, e.type,
                       (SELECT COUNT(*) FROM mention m WHERE m.entity_id = e.id) as cnt
                FROM entity e {type_clause}
                ORDER BY cnt DESC
                LIMIT ? OFFSET ?""",
            params,
        ).fetchall()
        return [Entity(id=r[0], canonical_name=r[1], type=r[2], mention_count=r[3]) for r in rows]

    # -- Resolution --

    def merge(self, keep_id: int, merge_id: int) -> Entity | None:
        """Merge merge_id into keep_id. Returns updated keep entity."""
        keep = self.get(keep_id)
        merge = self.get(merge_id)
        if not keep or not merge:
            return None

        # Move all mentions
        self._conn.execute(
            "UPDATE mention SET entity_id = ? WHERE entity_id = ?",
            (keep_id, merge_id),
        )

        # Move aliases
        for alias_row in self._conn.execute(
            "SELECT name FROM alias WHERE entity_id = ?", (merge_id,)
        ).fetchall():
            self._conn.execute(
                "INSERT OR IGNORE INTO alias (entity_id, name) VALUES (?, ?)",
                (keep_id, alias_row[0]),
            )

        # Add merged entity's canonical name as alias
        self._conn.execute(
            "INSERT OR IGNORE INTO alias (entity_id, name) VALUES (?, ?)",
            (keep_id, merge.canonical_name),
        )

        # Delete merged entity (CASCADE deletes its aliases)
        self._conn.execute("DELETE FROM entity WHERE id = ?", (merge_id,))
        self._conn.commit()

        return self.get(keep_id)

    def rename(self, entity_id: int, new_name: str) -> Entity | None:
        """Rename entity. Old name becomes an alias."""
        entity = self.get(entity_id)
        if not entity:
            return None

        # Add old name as alias
        self._conn.execute(
            "INSERT OR IGNORE INTO alias (entity_id, name) VALUES (?, ?)",
            (entity_id, entity.canonical_name),
        )

        self._conn.execute(
            "UPDATE entity SET canonical_name = ? WHERE id = ?",
            (new_name, entity_id),
        )

        # Remove alias if it matches new canonical name
        self._conn.execute(
            "DELETE FROM alias WHERE entity_id = ? AND name = ?",
            (entity_id, new_name),
        )

        self._conn.commit()
        return self.get(entity_id)

    def delete(self, entity_id: int) -> bool:
        """Delete entity and all its mentions."""
        cursor = self._conn.execute("DELETE FROM entity WHERE id = ?", (entity_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> dict:
        """Return counts by entity type and total mentions."""
        type_counts = {}
        for row in self._conn.execute(
            "SELECT type, COUNT(*) FROM entity GROUP BY type"
        ).fetchall():
            type_counts[row[0]] = row[1]

        total_mentions = self._conn.execute("SELECT COUNT(*) FROM mention").fetchone()[0]
        total_entities = self._conn.execute("SELECT COUNT(*) FROM entity").fetchone()[0]

        return {
            "total_entities": total_entities,
            "total_mentions": total_mentions,
            "by_type": type_counts,
        }

    def _cleanup_orphans(self):
        """Remove entities with zero mentions."""
        self._conn.execute(
            """DELETE FROM entity WHERE id NOT IN
               (SELECT DISTINCT entity_id FROM mention)"""
        )
        self._conn.commit()
