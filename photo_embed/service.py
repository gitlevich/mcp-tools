"""HTTP service daemon for photo-embed.

Owns the PhotoIndex, loaded models, and state. MCP servers connect as
thin clients. Only one instance should run at a time.

    uv run python service.py

Startup order:
    1. Write PID, start uvicorn  -- HTTP is up immediately
    2. Background thread: load index state (JSON + numpy)
    3. Background thread: load ONNX text encoders
    Handlers return {"loading": true} until the index is ready.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
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
_index_lock = threading.Lock()
_index_ready = threading.Event()
_refresh_lock = asyncio.Lock()


def _create_index() -> PhotoIndex:
    """Create the PhotoIndex (thread-safe, called from background thread)."""
    global _index
    with _index_lock:
        if _index is None:
            _index = PhotoIndex(CACHE_DIR)
    _index_ready.set()
    return _index


def get_index() -> PhotoIndex:
    """Return the PhotoIndex, blocking until it is loaded."""
    _index_ready.wait()
    return _index  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_LOADING = JSONResponse({"loading": True}, status_code=503)


async def search_page(request: Request) -> FileResponse:
    return FileResponse(_STATIC_DIR / "search.html")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "ready": _index_ready.is_set()})


async def status(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    data = get_index().status()
    # Include live activity/progress from reporter
    if reporter._current_operation:
        data["activity"] = {
            "operation": reporter._current_operation,
            "progress": reporter._current_progress,
        }
    return JSONResponse(data)


async def search(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    body = await request.json()
    query = body["query"]
    top_k = body.get("top_k", 10)

    with reporter.report("Searching photos"):
        results = await asyncio.to_thread(get_index().search, query, top_k)

    return JSONResponse([asdict(r) for r in results])


async def search_stream(request: Request) -> StreamingResponse:
    """SSE endpoint that streams search results progressively."""
    if not _index_ready.is_set():
        async def _loading():
            yield f"data: {json.dumps([])}\n\n"
        return StreamingResponse(
            _loading(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    body = await request.json()
    query = body["query"]
    top_k = body.get("top_k", 200)

    async def generate():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run():
            try:
                for batch in get_index().search_progressive(query, top_k):
                    loop.call_soon_threadsafe(queue.put_nowait, batch)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _run)

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            data = json.dumps([asdict(r) for r in item])
            yield f"data: {data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def refresh(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    async with _refresh_lock:
        with reporter.report("Refreshing index") as r:
            stats = await asyncio.to_thread(
                get_index().refresh, progress_callback=r.set_progress
            )
    return JSONResponse(stats)


async def reindex(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    async with _refresh_lock:
        with reporter.report("Reindexing photos") as r:
            stats = await asyncio.to_thread(
                get_index().full_reindex, progress_callback=r.set_progress, confirm=True
            )
    return JSONResponse(stats)


async def pick_and_add_folder(request: Request) -> PlainTextResponse:
    if not _index_ready.is_set():
        return PlainTextResponse("Service is loading, try again shortly.", status_code=503)
    from picker import pick_folder
    path = await asyncio.to_thread(pick_folder)
    if path is None:
        return PlainTextResponse("No folder selected (dialog was cancelled).")
    msg = get_index().add_folder(path)
    return PlainTextResponse(msg)


async def add_folder(request: Request) -> PlainTextResponse:
    if not _index_ready.is_set():
        return PlainTextResponse("Service is loading, try again shortly.", status_code=503)
    body = await request.json()
    path = body["path"]
    msg = get_index().add_folder(path)
    return PlainTextResponse(msg)


async def remove_folder(request: Request) -> PlainTextResponse:
    if not _index_ready.is_set():
        return PlainTextResponse("Service is loading, try again shortly.", status_code=503)
    body = await request.json()
    path = body["path"]
    msg = get_index().remove_folder(path)
    return PlainTextResponse(msg)


async def list_folders(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    return JSONResponse(get_index().list_folders())


async def annotate(request: Request) -> PlainTextResponse:
    if not _index_ready.is_set():
        return PlainTextResponse("Service is loading, try again shortly.", status_code=503)
    body = await request.json()
    msg = get_index().annotate(body["path"], body["text"])
    return PlainTextResponse(msg)


async def get_annotations(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    path = request.query_params["path"]
    return JSONResponse(get_index().get_annotations(path))


async def remove_annotation(request: Request) -> PlainTextResponse:
    if not _index_ready.is_set():
        return PlainTextResponse("Service is loading, try again shortly.", status_code=503)
    body = await request.json()
    msg = get_index().remove_annotation(body["path"], body["text"])
    return PlainTextResponse(msg)


async def pending_tasks(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    return JSONResponse(get_index().pending_tasks())


async def search_thoughts(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    body = await request.json()
    query = body["query"]
    top_k = body.get("top_k", 10)
    results = await asyncio.to_thread(get_index().search_notes, query, top_k)
    return JSONResponse([asdict(r) for r in results])


async def thought_status(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    data = get_index().status()
    return JSONResponse({"total_chunks": data.get("total_notes", 0)})


async def thought_reindex(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    async with _refresh_lock:
        with reporter.report("Reindexing notes") as r:
            stats = await asyncio.to_thread(
                get_index().reindex_notes, progress_callback=r.set_progress
            )
    return JSONResponse(stats)


async def prune(request: Request) -> JSONResponse:
    if not _index_ready.is_set():
        return _LOADING
    async with _refresh_lock:
        with reporter.report("Pruning deleted images"):
            stats = await asyncio.to_thread(get_index().prune_deleted)
    return JSONResponse(stats)


async def connect_photos(request: Request) -> PlainTextResponse:
    if not _index_ready.is_set():
        return PlainTextResponse("Service is loading, try again shortly.", status_code=503)
    body = await request.json()
    msg = get_index().connect_photos_library(body.get("library_path") or None)
    return PlainTextResponse(msg)


async def disconnect_photos(request: Request) -> PlainTextResponse:
    if not _index_ready.is_set():
        return PlainTextResponse("Service is loading, try again shortly.", status_code=503)
    msg = get_index().disconnect_photos_library()
    return PlainTextResponse(msg)


THUMBNAIL_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

routes = [
    Route("/", search_page, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
    Route("/status", status, methods=["GET"]),
    Route("/search", search, methods=["POST"]),
    Route("/search-stream", search_stream, methods=["POST"]),
    Route("/refresh", refresh, methods=["POST"]),
    Route("/reindex", reindex, methods=["POST"]),
    Route("/pick-folder", pick_and_add_folder, methods=["POST"]),
    Route("/add-folder", add_folder, methods=["POST"]),
    Route("/remove-folder", remove_folder, methods=["POST"]),
    Route("/folders", list_folders, methods=["GET"]),
    Route("/annotate", annotate, methods=["POST"]),
    Route("/annotations", get_annotations, methods=["GET"]),
    Route("/remove-annotation", remove_annotation, methods=["POST"]),
    Route("/search-thoughts", search_thoughts, methods=["POST"]),
    Route("/thought-status", thought_status, methods=["GET"]),
    Route("/search-thoughts-reindex", thought_reindex, methods=["POST"]),
    Route("/prune", prune, methods=["POST"]),
    Route("/tasks", pending_tasks, methods=["GET"]),
    Route("/connect-photos", connect_photos, methods=["POST"]),
    Route("/disconnect-photos", disconnect_photos, methods=["POST"]),
    Mount("/thumbnails", StaticFiles(directory=str(THUMBNAIL_PREVIEW_DIR)), name="thumbnails"),
]

app = Starlette(routes=routes)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _set_process_priority() -> None:
    """Set low process priority via nice. Applied before uvicorn starts.

    Does NOT set macOS QoS BACKGROUND — that starves the event loop thread
    and makes HTTP endpoints unresponsive during heavy computation.
    nice(15) is sufficient for being a good CPU citizen.
    """
    try:
        os.nice(NICE_VALUE)
        logger.info("Set nice value to %d", NICE_VALUE)
    except OSError:
        logger.debug("Could not set nice value")


def _write_pid() -> None:
    SERVICE_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    SERVICE_PID_FILE.write_text(str(os.getpid()))
    logger.info("PID file: %s", SERVICE_PID_FILE)


def _cleanup_pid(*_args) -> None:
    SERVICE_PID_FILE.unlink(missing_ok=True)
    reporter.cleanup()


def _background_startup() -> None:
    """Load index state and text encoders without blocking the event loop.

    Uvicorn is already listening. Handlers return 503 until _index_ready is set.
    """
    def _load():
        try:
            index = _create_index()
            logger.info(
                "Index loaded: %d images, %d notes",
                len(index._state.images), len(index._state.notes),
            )
            for name in index._text_capable_models():
                index._ensure_text_encoder(name)
            logger.info("Text encoder warmup complete")
        except Exception:
            logger.warning("Background startup failed", exc_info=True)

    threading.Thread(target=_load, name="background-startup", daemon=True).start()


if __name__ == "__main__":
    import atexit

    import uvicorn

    _write_pid()
    atexit.register(_cleanup_pid)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    _set_process_priority()

    # Index + encoders load in a background thread.
    # HTTP is up immediately; handlers return 503 until ready.
    _background_startup()

    logger.info("Starting photo-embed service on %s:%d", SERVICE_HOST, SERVICE_PORT)
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_level="warning")
