"""Thin MCP server for photo-embed.

Delegates all work to the photo-embed service daemon via HTTP.
No torch, numpy, or PIL imports in this process.
"""

import logging

from mcp.server.fastmcp import FastMCP

from client import ServiceClient
from config import DEFAULT_TOP_K
from picker import pick_folder as _pick_folder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("photo-embed")
client = ServiceClient()


@mcp.tool()
def search(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Search photos by natural language description.

    Uses multiple vision-language models (CLIP, SigLIP) to find photos
    matching the query text. Returns ranked results with file paths,
    similarity scores, and preview thumbnail paths.

    Args:
        query: Natural language description of the photo to find.
        top_k: Number of results to return (default 10).
    """
    return client.search(query, top_k)


@mcp.tool()
def find_similar(path: str, top_k: int = 20) -> str:
    """Find images visually similar to a given image.

    Uses pre-computed embeddings to find photos that look like the
    specified image. Fast — no model loading required.

    Args:
        path: Absolute path to the source image (must be in the index).
        top_k: Number of results to return (default 20).
    """
    return client.find_similar(path, top_k)


@mcp.tool()
def annotate(path: str, text: str) -> str:
    """Add a text annotation to a photo.

    Annotations are used during search to find photos by personal context,
    names, events, or any description you provide.

    Args:
        path: Absolute path to the image file.
        text: Annotation text (e.g., "my father at the beach", "graduation 2020").
    """
    return client.annotate(path, text)


@mcp.tool()
def get_annotations(path: str) -> str:
    """Get all annotations for a photo.

    Args:
        path: Absolute path to the image file.
    """
    annotations = client.get_annotations(path)
    if not annotations:
        return "No annotations for this image."
    return "\n".join(f"- {a}" for a in annotations)


@mcp.tool()
def remove_annotation(path: str, text: str) -> str:
    """Remove a specific annotation from a photo.

    Args:
        path: Absolute path to the image file.
        text: The exact annotation text to remove.
    """
    return client.remove_annotation(path, text)


@mcp.tool()
def add_folder(path: str) -> str:
    """Add a folder of photos to the index.

    The folder will be recursively scanned for JPEG, PNG, and HEIF images.
    New images are indexed on the next search or explicit refresh.

    Args:
        path: Absolute path to a folder containing photos.
    """
    return client.add_folder(path)


@mcp.tool()
def pick_folder() -> str:
    """Open a native macOS Finder dialog to select a photo folder.

    Launches a Finder file picker window. The selected folder is
    automatically added to the index.
    """
    path = _pick_folder()
    if path is None:
        return "No folder selected (dialog was cancelled)."
    return client.add_folder(path)


@mcp.tool()
def remove_folder(path: str) -> str:
    """Remove a folder from the index.

    All images from this folder are removed from the index.

    Args:
        path: Path to the folder to remove.
    """
    return client.remove_folder(path)


@mcp.tool()
def list_folders() -> str:
    """List all configured photo folders."""
    folders = client.list_folders()
    if not folders:
        return "No folders configured. Use add_folder to add one."
    return "\n".join(folders)


@mcp.tool()
def connect_photos_library(library_path: str = "") -> str:
    """Connect to an Apple Photos library for indexing.

    Reads photos directly from the Photos library database.
    Requires Full Disk Access for the server process.
    Only images present on disk are indexed (iCloud-only photos are skipped).

    Args:
        library_path: Path to .photoslibrary bundle. Empty string uses the default library.
    """
    return client.connect_photos_library(library_path or None)


@mcp.tool()
def disconnect_photos_library() -> str:
    """Disconnect the Apple Photos library.

    Removes all Photos library images from the index.
    """
    return client.disconnect_photos_library()


@mcp.tool()
def index_status() -> str:
    """Report the current state of the photo index.

    Returns configured folders, Photos library connection, enabled models,
    total indexed images, and embedding counts per model.
    """
    s = client.status()
    lines = [
        f"Folders: {len(s['folders'])}",
        *[f"  {f}" for f in s["folders"]],
        f"Photos library: {s['photos_library'] or 'not connected'}",
        f"Models: {', '.join(s['enabled_models'])}",
        f"Total images: {s['total_images']}",
        f"Annotated images: {s['annotated_images']}",
        f"Embeddings per model:",
        *[f"  {m}: {c}" for m, c in s["embeddings"].items()],
        f"Last scan: {s['last_scan'] or 'never'}",
    ]
    return "\n".join(lines)


@mcp.tool()
def pending_tasks() -> str:
    """List photos with pending @task annotations.

    Returns tasks added by the user through the search UI.
    Each task has a path and the task text. Resolve tasks by
    removing the @task annotation and adding a result annotation.
    """
    tasks = client.pending_tasks()
    if not tasks:
        return "No pending tasks."
    lines = []
    for t in tasks:
        lines.append(f"- {t['path']}\n  {t['task']}")
    return "\n".join(lines)


@mcp.tool()
def refresh() -> str:
    """Refresh the photo index.

    Scans configured folders for new, changed, or deleted photos and updates
    the index incrementally. Much faster than a full reindex.
    """
    stats = client.refresh()
    parts = []
    if stats.get("new"):
        parts.append(f"+{stats['new']} new")
    if stats.get("changed"):
        parts.append(f"~{stats['changed']} changed")
    if stats.get("deleted"):
        parts.append(f"-{stats['deleted']} deleted")
    if not parts:
        return f"Index is up to date ({stats.get('total', 0)} images)."
    return f"Refreshed: {', '.join(parts)} ({stats.get('total', 0)} total, {stats.get('elapsed', 0):.1f}s)."


@mcp.tool()
def reindex() -> str:
    """Force a full re-index of all configured folders.

    Drops existing embeddings and thumbnails, then rebuilds from scratch.
    This can be slow for large photo libraries.
    """
    return client.reindex()


@mcp.tool()
def refresh_faces() -> str:
    """Detect faces in photos that haven't been scanned yet.

    Runs face detection (RetinaFace) and face embedding (ArcFace) on all
    indexed images that haven't been processed for faces. Incremental --
    skips already-scanned images.
    """
    stats = client.refresh_faces()
    return (
        f"Scanned {stats['images_scanned']} images, "
        f"found {stats['new_faces']} new faces "
        f"({stats['total_faces']} total)"
    )


@mcp.tool()
def label_face(path: str, face_idx: int, label: str) -> str:
    """Label a face in a photo with a person's name.

    After labeling, the person's name becomes searchable via text search
    (adds an @person annotation to the image).

    Args:
        path: Absolute path to the image file.
        face_idx: Index of the face within the image (from get_faces).
        label: Person's name.
    """
    return client.label_face(path, face_idx, label)


@mcp.tool()
def find_person(path: str, face_idx: int = 0, top_k: int = 50) -> str:
    """Find all photos containing the same person as a face in the given photo.

    Uses pre-computed ArcFace embeddings for face matching. Returns ranked
    results with similarity scores, same format as search results.

    Args:
        path: Absolute path to the image containing the reference face.
        face_idx: Index of the face in that image (default 0 for single-face images).
        top_k: Number of results to return (default 50).
    """
    return client.find_person(path, face_idx, top_k)


if __name__ == "__main__":
    mcp.run(transport="stdio")
