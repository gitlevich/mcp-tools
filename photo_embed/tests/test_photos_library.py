import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from photos_library import scan_library, SUPPORTED_UTIS


def _create_photos_db(db_path: Path, rows: list[tuple]) -> None:
    """Create a minimal Photos.sqlite with the columns scan_library queries."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE ZASSET (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZDIRECTORY TEXT,
            ZFILENAME TEXT,
            ZCLOUDBATCHPUBLISHDATE REAL,
            ZDATECREATED REAL,
            ZLATITUDE REAL,
            ZLONGITUDE REAL,
            ZTRASHEDSTATE INTEGER DEFAULT 0,
            ZKIND INTEGER DEFAULT 0,
            ZCOMPLETE INTEGER DEFAULT 1,
            ZUNIFORMTYPEIDENTIFIER TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE ZADDITIONALASSETATTRIBUTES (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZASSET INTEGER,
            ZTITLE TEXT,
            ZORIGINALFILENAME TEXT
        )
    """)
    for row in rows:
        # row: (directory, filename, cloud_batch_date, trashed, kind, complete, uti,
        #       date_created, title, original_filename, latitude, longitude)
        # Short rows (7 fields) use the old format for backward compat
        if len(row) == 7:
            directory, filename, cbd, trashed, kind, complete, uti = row
            date_created, title, orig_fn, lat, lon = None, None, None, None, None
        elif len(row) == 10:
            directory, filename, cbd, trashed, kind, complete, uti, date_created, title, orig_fn = row
            lat, lon = None, None
        else:
            directory, filename, cbd, trashed, kind, complete, uti, date_created, title, orig_fn, lat, lon = row

        cursor = conn.execute(
            "INSERT INTO ZASSET (ZDIRECTORY, ZFILENAME, ZCLOUDBATCHPUBLISHDATE, "
            "ZDATECREATED, ZLATITUDE, ZLONGITUDE, ZTRASHEDSTATE, ZKIND, ZCOMPLETE, "
            "ZUNIFORMTYPEIDENTIFIER) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (directory, filename, cbd, date_created, lat, lon, trashed, kind, complete, uti),
        )
        asset_pk = cursor.lastrowid
        conn.execute(
            "INSERT INTO ZADDITIONALASSETATTRIBUTES (ZASSET, ZTITLE, ZORIGINALFILENAME) "
            "VALUES (?, ?, ?)",
            (asset_pk, title, orig_fn),
        )
    conn.commit()
    conn.close()


def _add_people_tables(db_path: Path, people: list[tuple[int, str]]) -> None:
    """Add ZPERSON and ZDETECTEDFACE tables with face-asset links."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ZPERSON (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZFULLNAME TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ZDETECTEDFACE (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZASSET INTEGER,
            ZPERSON INTEGER
        )
    """)
    for asset_pk, name in people:
        cursor = conn.execute("INSERT INTO ZPERSON (ZFULLNAME) VALUES (?)", (name,))
        person_pk = cursor.lastrowid
        conn.execute(
            "INSERT INTO ZDETECTEDFACE (ZASSET, ZPERSON) VALUES (?, ?)",
            (asset_pk, person_pk),
        )
    conn.commit()
    conn.close()


def _add_keyword_tables(db_path: Path, keywords: list[tuple[int, str]]) -> None:
    """Add ZKEYWORD and Z_1KEYWORDS junction table."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ZKEYWORD (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZTITLE TEXT
        )
    """)
    # Use Z_40KEYWORDS as the FK column (one of the common versions)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Z_1KEYWORDS (
            Z_1ASSETATTRIBUTES INTEGER,
            Z_40KEYWORDS INTEGER
        )
    """)
    for asset_pk, keyword in keywords:
        cursor = conn.execute("INSERT INTO ZKEYWORD (ZTITLE) VALUES (?)", (keyword,))
        kw_pk = cursor.lastrowid
        # Z_1ASSETATTRIBUTES points to ZAddAttr.Z_PK, which equals asset_pk
        # in our simple schema (1:1 with same PK)
        conn.execute(
            "INSERT INTO Z_1KEYWORDS (Z_1ASSETATTRIBUTES, Z_40KEYWORDS) VALUES (?, ?)",
            (asset_pk, kw_pk),
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def photos_lib(tmp_path):
    """Create a fake Photos library structure."""
    lib = tmp_path / "Test.photoslibrary"
    (lib / "database").mkdir(parents=True)
    (lib / "originals" / "A").mkdir(parents=True)
    (lib / "scopes" / "cloudsharing" / "data" / "B").mkdir(parents=True)
    return lib


def test_scan_regular_photo(photos_lib):
    (photos_lib / "originals" / "A" / "photo.jpg").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "photo.jpg", None, 0, 0, 1, "public.jpeg"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 1
    assert paths[0] == photos_lib / "originals" / "A" / "photo.jpg"


def test_scan_shared_photo(photos_lib):
    (photos_lib / "scopes" / "cloudsharing" / "data" / "B" / "shared.heic").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("B", "shared.heic", 12345.0, 0, 0, 1, "public.heic"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 1
    assert paths[0] == photos_lib / "scopes" / "cloudsharing" / "data" / "B" / "shared.heic"


def test_scan_absolute_path(photos_lib, tmp_path):
    abs_dir = tmp_path / "external"
    abs_dir.mkdir()
    (abs_dir / "ext.png").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        (str(abs_dir), "ext.png", None, 0, 0, 1, "public.png"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 1
    assert paths[0] == abs_dir / "ext.png"


def test_skips_missing_files(photos_lib):
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "gone.jpg", None, 0, 0, 1, "public.jpeg"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 0


def test_skips_trashed(photos_lib):
    (photos_lib / "originals" / "A" / "trashed.jpg").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "trashed.jpg", None, 1, 0, 1, "public.jpeg"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 0


def test_skips_videos(photos_lib):
    (photos_lib / "originals" / "A" / "video.mov").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "video.mov", None, 0, 1, 1, "com.apple.quicktime-movie"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 0


def test_skips_incomplete(photos_lib):
    (photos_lib / "originals" / "A" / "partial.jpg").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "partial.jpg", None, 0, 0, 0, "public.jpeg"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 0


def test_handles_null_directory(photos_lib):
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        (None, "orphan.jpg", None, 0, 0, 1, "public.jpeg"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 0


def test_missing_database(tmp_path):
    lib = tmp_path / "Empty.photoslibrary"
    lib.mkdir()

    paths, meta = scan_library(lib)
    assert paths == []
    assert meta == {}


def test_multiple_photos(photos_lib):
    (photos_lib / "originals" / "A" / "one.jpg").touch()
    (photos_lib / "originals" / "A" / "two.heic").touch()
    (photos_lib / "scopes" / "cloudsharing" / "data" / "B" / "three.png").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "one.jpg", None, 0, 0, 1, "public.jpeg"),
        ("A", "two.heic", None, 0, 0, 1, "public.heic"),
        ("B", "three.png", 99.0, 0, 0, 1, "public.png"),
        ("A", "missing.jpg", None, 0, 0, 1, "public.jpeg"),
    ])

    paths, meta = scan_library(photos_lib)
    names = {p.name for p in paths}
    assert names == {"one.jpg", "two.heic", "three.png"}


# -- metadata extraction --

def test_title_and_original_filename(photos_lib):
    (photos_lib / "originals" / "A" / "img.jpg").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "img.jpg", None, 0, 0, 1, "public.jpeg",
         None, "Sunset at the beach", "DSC_1234.jpg"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 1
    resolved = str(paths[0].resolve())
    assert resolved in meta
    assert meta[resolved]["title"] == "Sunset at the beach"
    assert meta[resolved]["original_filename"] == "DSC_1234.jpg"


def test_date_extraction(photos_lib):
    (photos_lib / "originals" / "A" / "dated.jpg").touch()
    # Apple epoch for 2023-07-15 12:00:00 UTC = Unix - 978307200
    apple_ts = 1689422400.0 - 978307200
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "dated.jpg", None, 0, 0, 1, "public.jpeg",
         apple_ts, None, None),
    ])

    paths, meta = scan_library(photos_lib)
    resolved = str(paths[0].resolve())
    assert "date" in meta[resolved]
    assert "2023" in meta[resolved]["date"]


def test_people_names(photos_lib):
    (photos_lib / "originals" / "A" / "family.jpg").touch()
    db_path = photos_lib / "database" / "Photos.sqlite"
    _create_photos_db(db_path, [
        ("A", "family.jpg", None, 0, 0, 1, "public.jpeg"),
    ])
    _add_people_tables(db_path, [(1, "Alice"), (1, "Bob")])

    paths, meta = scan_library(photos_lib)
    resolved = str(paths[0].resolve())
    assert set(meta[resolved]["people"]) == {"Alice", "Bob"}


def test_keywords(photos_lib):
    (photos_lib / "originals" / "A" / "tagged.jpg").touch()
    db_path = photos_lib / "database" / "Photos.sqlite"
    _create_photos_db(db_path, [
        ("A", "tagged.jpg", None, 0, 0, 1, "public.jpeg"),
    ])
    _add_keyword_tables(db_path, [(1, "vacation"), (1, "summer")])

    paths, meta = scan_library(photos_lib)
    resolved = str(paths[0].resolve())
    assert set(meta[resolved]["keywords"]) == {"vacation", "summer"}


def test_no_metadata_tables_graceful(photos_lib):
    """People/keyword queries should not fail if those tables don't exist."""
    (photos_lib / "originals" / "A" / "plain.jpg").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "plain.jpg", None, 0, 0, 1, "public.jpeg"),
    ])

    paths, meta = scan_library(photos_lib)
    assert len(paths) == 1
    # No title, no date, no people, no keywords — metadata may be empty
    resolved = str(paths[0].resolve())
    if resolved in meta:
        assert "people" not in meta[resolved]
        assert "keywords" not in meta[resolved]


# -- GPS extraction --

def test_gps_included_in_metadata(photos_lib):
    (photos_lib / "originals" / "A" / "geo.jpg").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "geo.jpg", None, 0, 0, 1, "public.jpeg",
         None, None, None, 48.8566, 2.3522),
    ])

    paths, meta = scan_library(photos_lib)
    resolved = str(paths[0].resolve())
    assert meta[resolved]["latitude"] == 48.8566
    assert meta[resolved]["longitude"] == 2.3522


def test_gps_null_island_filtered(photos_lib):
    (photos_lib / "originals" / "A" / "null.jpg").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "null.jpg", None, 0, 0, 1, "public.jpeg",
         None, None, None, 0, 0),
    ])

    paths, meta = scan_library(photos_lib)
    resolved = str(paths[0].resolve())
    if resolved in meta:
        assert "latitude" not in meta[resolved]


def test_gps_none_excluded(photos_lib):
    (photos_lib / "originals" / "A" / "nogps.jpg").touch()
    _create_photos_db(photos_lib / "database" / "Photos.sqlite", [
        ("A", "nogps.jpg", None, 0, 0, 1, "public.jpeg",
         None, None, None, None, None),
    ])

    paths, meta = scan_library(photos_lib)
    resolved = str(paths[0].resolve())
    if resolved in meta:
        assert "latitude" not in meta[resolved]
