"""HTTP service daemon for photo-embed.

Owns the PhotoIndex, loaded models, and state. MCP servers connect as
thin clients. Only one instance should run at a time.

    uv run python service.py
"""

import asyncio
import ctypes
import ctypes.util
import json
import logging
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity import ActivityReporter

from config import (
    CACHE_DIR,
    NICE_VALUE,
    SERVICE_HOST,
    SERVICE_PID_FILE,
    SERVICE_PORT,
    THUMBNAIL_PREVIEW_DIR,
)
from indexer import PhotoIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

reporter = ActivityReporter("photo-embed")
_index: PhotoIndex | None = None
_refresh_lock = asyncio.Lock()


def get_index() -> PhotoIndex:
    global _index
    if _index is None:
        _index = PhotoIndex(CACHE_DIR)
    return _index


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


_STATIC_DIR = Path(__file__).resolve().parent / "static"


async def search_page(request: Request) -> FileResponse:
    return FileResponse(_STATIC_DIR / "search.html")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


async def status(request: Request) -> JSONResponse:
    data = get_index().status()
    # Include live activity/progress from reporter
    if reporter._current_operation:
        data["activity"] = {
            "operation": reporter._current_operation,
            "progress": reporter._current_progress,
        }
    return JSONResponse(data)


async def search(request: Request) -> JSONResponse:
    body = await request.json()
    query = body["query"]
    top_k = body.get("top_k", 10)

    with reporter.report("Searching photos"):
        results = await asyncio.to_thread(get_index().search, query, top_k)

    return JSONResponse([asdict(r) for r in results])


async def refresh(request: Request) -> JSONResponse:
    async with _refresh_lock:
        with reporter.report("Refreshing index") as r:
            stats = await asyncio.to_thread(
                get_index().refresh, progress_callback=r.set_progress
            )
    return JSONResponse(stats)


async def reindex(request: Request) -> JSONResponse:
    async with _refresh_lock:
        with reporter.report("Reindexing photos") as r:
            stats = await asyncio.to_thread(
                get_index().full_reindex, progress_callback=r.set_progress
            )
    return JSONResponse(stats)


async def pick_and_add_folder(request: Request) -> PlainTextResponse:
    from picker import pick_folder
    path = await asyncio.to_thread(pick_folder)
    if path is None:
        return PlainTextResponse("No folder selected (dialog was cancelled).")
    msg = get_index().add_folder(path)
    return PlainTextResponse(msg)


async def add_folder(request: Request) -> PlainTextResponse:
    body = await request.json()
    msg = get_index().add_folder(body["path"])
    return PlainTextResponse(msg)


async def remove_folder(request: Request) -> PlainTextResponse:
    body = await request.json()
    msg = get_index().remove_folder(body["path"])
    return PlainTextResponse(msg)


async def list_folders(request: Request) -> JSONResponse:
    return JSONResponse(get_index().list_folders())


async def annotate(request: Request) -> PlainTextResponse:
    body = await request.json()
    msg = get_index().annotate(body["path"], body["text"])
    return PlainTextResponse(msg)


async def get_annotations(request: Request) -> JSONResponse:
    path = request.query_params["path"]
    return JSONResponse(get_index().get_annotations(path))


async def remove_annotation(request: Request) -> PlainTextResponse:
    body = await request.json()
    msg = get_index().remove_annotation(body["path"], body["text"])
    return PlainTextResponse(msg)


async def pending_tasks(request: Request) -> JSONResponse:
    return JSONResponse(get_index().pending_tasks())


async def connect_photos(request: Request) -> PlainTextResponse:
    body = await request.json()
    msg = get_index().connect_photos_library(body.get("library_path") or None)
    return PlainTextResponse(msg)


async def disconnect_photos(request: Request) -> PlainTextResponse:
    msg = get_index().disconnect_photos_library()
    return PlainTextResponse(msg)


THUMBNAIL_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

routes = [
    Route("/", search_page, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
    Route("/status", status, methods=["GET"]),
    Route("/search", search, methods=["POST"]),
    Route("/refresh", refresh, methods=["POST"]),
    Route("/reindex", reindex, methods=["POST"]),
    Route("/pick-folder", pick_and_add_folder, methods=["POST"]),
    Route("/add-folder", add_folder, methods=["POST"]),
    Route("/remove-folder", remove_folder, methods=["POST"]),
    Route("/folders", list_folders, methods=["GET"]),
    Route("/annotate", annotate, methods=["POST"]),
    Route("/annotations", get_annotations, methods=["GET"]),
    Route("/remove-annotation", remove_annotation, methods=["POST"]),
    Route("/tasks", pending_tasks, methods=["GET"]),
    Route("/connect-photos", connect_photos, methods=["POST"]),
    Route("/disconnect-photos", disconnect_photos, methods=["POST"]),
    Mount("/thumbnails", StaticFiles(directory=str(THUMBNAIL_PREVIEW_DIR)), name="thumbnails"),
]

app = Starlette(routes=routes)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _set_background_priority() -> None:
    try:
        os.nice(NICE_VALUE)
        logger.info("Set nice value to %d", NICE_VALUE)
    except OSError:
        logger.debug("Could not set nice value")

    try:
        lib = ctypes.CDLL(ctypes.util.find_library("System"))
        QOS_CLASS_BACKGROUND = 0x09
        lib.pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0)
        logger.info("Set macOS QoS to BACKGROUND")
    except Exception:
        logger.debug("Could not set macOS QoS class")


def _write_pid() -> None:
    SERVICE_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    SERVICE_PID_FILE.write_text(str(os.getpid()))
    logger.info("PID file: %s", SERVICE_PID_FILE)


def _cleanup_pid(*_args) -> None:
    SERVICE_PID_FILE.unlink(missing_ok=True)
    reporter.cleanup()


if __name__ == "__main__":
    import atexit

    import uvicorn

    _set_background_priority()
    _write_pid()
    atexit.register(_cleanup_pid)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    logger.info("Starting photo-embed service on %s:%d", SERVICE_HOST, SERVICE_PORT)
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_level="warning")
