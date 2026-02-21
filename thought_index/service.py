"""HTTP service daemon for thought-index.

Owns the ThoughtIndex and serves search over HTTP.
Only one instance should run at a time.

    uv run python service.py
"""

import asyncio
import ctypes
import ctypes.util
import logging
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity import ActivityReporter

from config import (
    CHROMA_DIR,
    DEFAULT_TOP_K,
    SERVICE_HOST,
    SERVICE_PID_FILE,
    SERVICE_PORT,
)
from indexer import ThoughtIndex
from sources import add_source, load_sources, remove_source

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

reporter = ActivityReporter("thought-index")
_index: ThoughtIndex | None = None


def get_index() -> ThoughtIndex:
    global _index
    if _index is None:
        _index = ThoughtIndex(CHROMA_DIR)
    return _index


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


async def status(request: Request) -> JSONResponse:
    index = get_index()
    sources = load_sources()
    data = {
        "total_chunks": index._collection.count(),
        "sources": [{"label": s.label, "path": str(s.path)} for s in sources],
    }
    if reporter._current_operation:
        data["activity"] = {
            "operation": reporter._current_operation,
            "progress": reporter._current_progress,
        }
    return JSONResponse(data)


async def search(request: Request) -> JSONResponse:
    body = await request.json()
    query = body["query"]
    top_k = body.get("top_k", DEFAULT_TOP_K)

    with reporter.report("Searching thoughts") as r:
        index = get_index()
        index.schedule_refresh(progress_callback=r.set_progress)
        results = await asyncio.to_thread(index.search, query, top_k)

    return JSONResponse([asdict(r) for r in results])


async def add_source_folder(request: Request) -> JSONResponse:
    body = await request.json()
    path = body.get("path", "").strip()
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    try:
        source = add_source(path)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    with reporter.report("Indexing new source") as r:
        stats = await asyncio.to_thread(
            get_index().index_source, source, progress_callback=r.set_progress
        )
    return JSONResponse({"label": source.label, "path": source.path, **stats})


async def remove_source_folder(request: Request) -> JSONResponse:
    body = await request.json()
    path = body.get("path", "").strip()
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    sources = load_sources()
    source = next((s for s in sources if s.path == path), None)
    if source is None:
        return JSONResponse({"error": f"Not tracked: {path}"}, status_code=404)
    try:
        remove_source(source.label)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    removed = get_index().remove_source_chunks(source.label)
    return JSONResponse({"removed_chunks": removed})


async def list_sources(request: Request) -> JSONResponse:
    sources = load_sources()
    return JSONResponse([{"label": s.label, "path": s.path} for s in sources])


async def do_reindex(request: Request) -> JSONResponse:
    with reporter.report("Reindexing thoughts") as r:
        stats = await asyncio.to_thread(
            get_index().full_reindex, progress_callback=r.set_progress
        )
    return JSONResponse(stats)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

routes = [
    Route("/health", health, methods=["GET"]),
    Route("/status", status, methods=["GET"]),
    Route("/search", search, methods=["POST"]),
    Route("/add-source", add_source_folder, methods=["POST"]),
    Route("/remove-source", remove_source_folder, methods=["POST"]),
    Route("/sources", list_sources, methods=["GET"]),
    Route("/reindex", do_reindex, methods=["POST"]),
]

app = Starlette(routes=routes)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

NICE_VALUE = 15


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

    logger.info("Starting thought-index service on %s:%d", SERVICE_HOST, SERVICE_PORT)
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_level="warning")
