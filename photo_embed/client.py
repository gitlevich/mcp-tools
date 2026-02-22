"""HTTP client for the photo-embed service daemon.

Auto-launches the service if it's not running.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from config import (
    SERVICE_HOST,
    SERVICE_PID_FILE,
    SERVICE_PORT,
    SERVICE_STARTUP_TIMEOUT,
)

logger = logging.getLogger(__name__)


class ServiceClient:
    def __init__(self, base_url: str | None = None):
        self._base_url = base_url or f"http://{SERVICE_HOST}:{SERVICE_PORT}"
        self._http = httpx.Client(base_url=self._base_url, timeout=600)

    def _is_alive(self) -> bool:
        try:
            resp = self._http.get("/health", timeout=2)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def _ensure_service(self) -> None:
        if self._is_alive():
            return

        logger.info("Service not running, launching...")
        service_script = Path(__file__).resolve().parent / "service.py"
        subprocess.Popen(
            [sys.executable, str(service_script)],
            cwd=str(service_script.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        deadline = time.time() + SERVICE_STARTUP_TIMEOUT
        while time.time() < deadline:
            time.sleep(0.5)
            if self._is_alive():
                logger.info("Service is ready")
                return

        raise RuntimeError(
            f"Service did not start within {SERVICE_STARTUP_TIMEOUT}s"
        )

    def _post(self, path: str, json: dict | None = None) -> httpx.Response:
        self._ensure_service()
        return self._http.post(path, json=json or {})

    def _get(self, path: str, params: dict | None = None) -> httpx.Response:
        self._ensure_service()
        return self._http.get(path, params=params)

    # -- tool methods --

    def search(self, query: str, top_k: int = 10) -> str:
        resp = self._post("/search", {"query": query, "top_k": top_k})
        results = resp.json()
        if not results:
            return "No matching photos found."
        import json
        return json.dumps(results, indent=2)

    def find_similar(self, path: str, top_k: int = 20) -> str:
        resp = self._post("/find-similar", {"path": path, "top_k": top_k})
        results = resp.json()
        if not results:
            return "No similar images found."
        import json
        return json.dumps(results, indent=2)

    def refresh(self) -> dict:
        return self._post("/refresh").json()

    def reindex(self) -> str:
        stats = self._post("/reindex").json()
        return (
            f"Re-indexed {stats['images']} images "
            f"with models [{', '.join(stats['models'])}] "
            f"in {stats['elapsed']:.1f}s."
        )

    def add_folder(self, path: str) -> str:
        return self._post("/add-folder", {"path": path}).text

    def remove_folder(self, path: str) -> str:
        return self._post("/remove-folder", {"path": path}).text

    def list_folders(self) -> list[str]:
        return self._get("/folders").json()

    def annotate(self, path: str, text: str) -> str:
        return self._post("/annotate", {"path": path, "text": text}).text

    def get_annotations(self, path: str) -> list[str]:
        return self._get("/annotations", {"path": path}).json()

    def remove_annotation(self, path: str, text: str) -> str:
        return self._post("/remove-annotation", {"path": path, "text": text}).text

    def connect_photos_library(self, library_path: str | None = None) -> str:
        return self._post("/connect-photos", {"library_path": library_path or ""}).text

    def disconnect_photos_library(self) -> str:
        return self._post("/disconnect-photos").text

    def pending_tasks(self) -> list[dict]:
        return self._get("/tasks").json()

    def status(self) -> dict:
        return self._get("/status").json()

    def health(self) -> bool:
        return self._is_alive()

    # -- face operations --

    def refresh_faces(self) -> dict:
        return self._post("/refresh-faces").json()

    def get_faces(self, path: str) -> list[dict]:
        return self._get("/faces", {"path": path}).json()

    def label_face(self, path: str, face_idx: int, label: str) -> str:
        return self._post(
            "/label-face", {"path": path, "face_idx": face_idx, "label": label}
        ).text

    def find_person(self, path: str, face_idx: int = 0, top_k: int = 50) -> str:
        import json

        results = self._post(
            "/find-person", {"path": path, "face_idx": face_idx, "top_k": top_k}
        ).json()
        if not results:
            return "No matching faces found."
        return json.dumps(results, indent=2)

    def batch_label(self, matches: list[dict], label: str) -> dict:
        return self._post("/batch-label", {"matches": matches, "label": label}).json()

    def get_people(self) -> list[dict]:
        return self._get("/people").json()

    # -- entity operations --

    def extract_entities(self) -> dict:
        return self._post("/extract-entities").json()

    def search_entities(self, query: str, entity_type: str | None = None, limit: int = 20) -> list[dict]:
        body: dict = {"query": query, "limit": limit}
        if entity_type:
            body["type"] = entity_type
        return self._post("/search-entities", body).json()

    def get_entity(self, entity_id: int) -> dict:
        return self._get("/entity", {"id": str(entity_id)}).json()

    def list_entities(self, entity_type: str | None = None, limit: int = 50) -> list[dict]:
        params: dict = {"limit": str(limit)}
        if entity_type:
            params["type"] = entity_type
        return self._get("/entities", params).json()

    def merge_entities(self, keep_id: int, merge_id: int) -> dict:
        return self._post("/merge-entities", {"keep_id": keep_id, "merge_id": merge_id}).json()

    def rename_entity(self, entity_id: int, new_name: str) -> dict:
        return self._post("/rename-entity", {"entity_id": entity_id, "new_name": new_name}).json()

    def delete_entity(self, entity_id: int) -> dict:
        return self._post("/delete-entity", {"entity_id": entity_id}).json()

    def entity_stats(self) -> dict:
        return self._get("/entity-stats").json()
