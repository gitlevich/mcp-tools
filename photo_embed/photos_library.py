"""Read photos from an Apple Photos library via direct SQLite access."""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LIBRARY = Path.home() / "Pictures" / "Photos Library.photoslibrary"

SUPPORTED_UTIS = {
    "public.jpeg",
    "public.png",
    "public.heic",
    "public.heif",
    "public.tiff",
    "com.apple.quicktime-image",
}

# Apple epoch offset: seconds between 1970-01-01 and 2001-01-01
_APPLE_EPOCH_OFFSET = 978307200

_SCAN_QUERY = """
SELECT
    ZASSET.Z_PK,
    ZASSET.ZDIRECTORY,
    ZASSET.ZFILENAME,
    ZASSET.ZCLOUDBATCHPUBLISHDATE,
    ZASSET.ZDATECREATED,
    ZASSET.ZLATITUDE,
    ZASSET.ZLONGITUDE,
    ZAddAttr.ZTITLE,
    ZAddAttr.ZORIGINALFILENAME
FROM ZASSET
LEFT JOIN ZADDITIONALASSETATTRIBUTES AS ZAddAttr
    ON ZAddAttr.ZASSET = ZASSET.Z_PK
WHERE ZASSET.ZTRASHEDSTATE = 0
  AND ZASSET.ZKIND = 0
  AND ZASSET.ZCOMPLETE = 1
  AND ZASSET.ZUNIFORMTYPEIDENTIFIER IN ({placeholders})
"""


def list_libraries() -> list[Path]:
    """Find Photos libraries on this machine."""
    pictures = Path.home() / "Pictures"
    if not pictures.is_dir():
        return []
    return sorted(pictures.glob("*.photoslibrary"))


def _apple_epoch_to_str(ts: float | None) -> str:
    """Convert Apple epoch timestamp to 'YYYY:MM:DD HH:MM:SS' for _format_date()."""
    if ts is None:
        return ""
    try:
        dt = datetime.fromtimestamp(ts + _APPLE_EPOCH_OFFSET, tz=timezone.utc)
        return dt.strftime("%Y:%m:%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        return ""


def _detect_face_columns(conn: sqlite3.Connection) -> tuple[str, str] | None:
    """Detect which FK columns ZDETECTEDFACE uses (schema varies by version)."""
    try:
        info = conn.execute("PRAGMA table_info(ZDETECTEDFACE)").fetchall()
    except sqlite3.OperationalError:
        return None
    columns = {row[1] for row in info}
    if "ZASSETFORFACE" in columns and "ZPERSONFORFACE" in columns:
        return "ZASSETFORFACE", "ZPERSONFORFACE"
    if "ZASSET" in columns and "ZPERSON" in columns:
        return "ZASSET", "ZPERSON"
    return None


def _detect_keyword_column(conn: sqlite3.Connection) -> str | None:
    """Detect the Z_NNN keyword FK column in the junction table."""
    try:
        info = conn.execute("PRAGMA table_info(Z_1KEYWORDS)").fetchall()
    except sqlite3.OperationalError:
        return None
    for row in info:
        col_name = row[1]
        if col_name.startswith("Z_") and col_name.endswith("KEYWORDS"):
            return col_name
    return None


def _fetch_people(
    conn: sqlite3.Connection,
    asset_pks: set[int],
) -> dict[int, list[str]]:
    """Fetch person names per asset PK."""
    face_cols = _detect_face_columns(conn)
    if not face_cols:
        return {}

    asset_col, person_col = face_cols
    query = f"""
        SELECT DF.{asset_col}, ZPERSON.ZFULLNAME
        FROM ZDETECTEDFACE AS DF
        JOIN ZPERSON ON ZPERSON.Z_PK = DF.{person_col}
        WHERE DF.{asset_col} IS NOT NULL
          AND ZPERSON.ZFULLNAME IS NOT NULL
          AND ZPERSON.ZFULLNAME != ''
    """
    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.OperationalError:
        logger.debug("People query failed", exc_info=True)
        return {}

    result: dict[int, list[str]] = {}
    for asset_pk, name in rows:
        if asset_pk in asset_pks:
            result.setdefault(asset_pk, []).append(name)
    return result


def _fetch_keywords(
    conn: sqlite3.Connection,
    asset_pks: set[int],
) -> dict[int, list[str]]:
    """Fetch keywords per asset PK via the junction table."""
    kw_col = _detect_keyword_column(conn)
    if not kw_col:
        return {}

    # Junction table Z_1KEYWORDS links ZADDITIONALASSETATTRIBUTES → ZKEYWORD.
    # We need to map back to asset PK via ZAddAttr.ZASSET.
    query = f"""
        SELECT ZAddAttr.ZASSET, ZKEYWORD.ZTITLE
        FROM Z_1KEYWORDS AS KW
        JOIN ZADDITIONALASSETATTRIBUTES AS ZAddAttr
            ON ZAddAttr.Z_PK = KW.Z_1ASSETATTRIBUTES
        JOIN ZKEYWORD
            ON ZKEYWORD.Z_PK = KW.{kw_col}
        WHERE ZKEYWORD.ZTITLE IS NOT NULL
          AND ZKEYWORD.ZTITLE != ''
    """
    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.OperationalError:
        logger.debug("Keywords query failed", exc_info=True)
        return {}

    result: dict[int, list[str]] = {}
    for asset_pk, keyword in rows:
        if asset_pk in asset_pks:
            result.setdefault(asset_pk, []).append(keyword)
    return result


def scan_library(
    library_path: Path | None = None,
) -> tuple[list[Path], dict[str, dict]]:
    """Scan an Apple Photos library and return on-disk image paths with metadata.

    Returns:
        (paths, metadata_lookup) where metadata_lookup maps resolved path strings
        to dicts with keys: title, people, keywords, date, original_filename.
    """
    lib = Path(library_path) if library_path else DEFAULT_LIBRARY
    db_path = lib / "database" / "Photos.sqlite"

    if not db_path.exists():
        logger.warning("Photos database not found: %s", db_path)
        return [], {}

    logger.info("Scanning Photos library: %s", lib)

    placeholders = ", ".join("?" for _ in SUPPORTED_UTIS)
    query = _SCAN_QUERY.format(placeholders=placeholders)

    originals_root = lib / "originals"
    shared_root = lib / "scopes" / "cloudsharing" / "data"

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(query, list(SUPPORTED_UTIS)).fetchall()

        # Collect asset PKs for batch metadata queries
        asset_rows: list[tuple[int, Path, str | None, str | None, float | None]] = []
        asset_pks: set[int] = set()

        for zpk, directory, filename, cloud_batch_date, date_created, lat, lon, title, orig_filename in rows:
            if not directory or not filename:
                continue

            if directory.startswith("/"):
                path = Path(directory) / filename
            elif cloud_batch_date is not None:
                path = shared_root / directory / filename
            else:
                path = originals_root / directory / filename

            if path.exists():
                asset_rows.append((zpk, path, title, orig_filename, date_created, lat, lon))
                asset_pks.add(zpk)

        logger.info("Found %d on-disk photos", len(asset_rows))

        # Batch-fetch people and keywords
        people_by_pk = _fetch_people(conn, asset_pks)
        keywords_by_pk = _fetch_keywords(conn, asset_pks)
    finally:
        conn.close()

    paths: list[Path] = []
    metadata_lookup: dict[str, dict] = {}

    for zpk, path, title, orig_filename, date_created, lat, lon in asset_rows:
        paths.append(path)

        meta: dict = {}
        if title:
            meta["title"] = title
        if orig_filename:
            meta["original_filename"] = orig_filename
        people = people_by_pk.get(zpk, [])
        if people:
            meta["people"] = people
        keywords = keywords_by_pk.get(zpk, [])
        if keywords:
            meta["keywords"] = keywords
        date_str = _apple_epoch_to_str(date_created)
        if date_str:
            meta["date"] = date_str
        if lat is not None and lon is not None and not (lat == 0 and lon == 0):
            meta["latitude"] = lat
            meta["longitude"] = lon

        if meta:
            metadata_lookup[str(path.resolve())] = meta

    skipped = len(rows) - len(paths)
    if skipped:
        logger.info("Skipped %d missing or iCloud-only photos", skipped)

    logger.info("Returning %d photos with %d metadata entries",
                len(paths), len(metadata_lookup))
    return paths, metadata_lookup
