"""Shared activity reporter for MCP servers.

Each server writes its current state to a JSON file in a shared directory.
The dashboard reads these files for live monitoring.
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

ACTIVITY_DIR = Path("/tmp/mcp-tools")
MAX_RECENT = 20


class ActivityReporter:
    def __init__(self, server_name: str, activity_dir: Path = ACTIVITY_DIR):
        self._server_name = server_name
        self._activity_dir = activity_dir
        self._activity_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._activity_dir / f"{server_name}.json"
        self._pid = os.getpid()
        self._recent: list[dict] = []
        self._current_operation: str | None = None
        self._current_started: float | None = None
        self._current_progress: dict | None = None
        self._active = True

        # If another live instance already owns this file, become a no-op
        # so dashboard-spawned temp servers don't clobber real activity data.
        if self._file.exists():
            try:
                existing = json.loads(self._file.read_text())
                existing_pid = existing.get("pid")
                if existing_pid and existing_pid != self._pid:
                    try:
                        os.kill(existing_pid, 0)
                    except ProcessLookupError:
                        pass  # Dead process, take over
                    except OSError:
                        # PermissionError etc — process is alive
                        self._active = False
                        return
                    else:
                        self._active = False
                        return
            except (json.JSONDecodeError, OSError):
                pass

        self._write_state()

    @contextmanager
    def report(self, operation: str):
        """Mark the server busy for the duration of the wrapped block."""
        self._current_operation = operation
        self._current_started = time.time()
        self._current_progress = None
        self._write_state()
        try:
            yield self
        finally:
            finished = time.time()
            entry = {
                "operation": operation,
                "started_at": self._current_started,
                "finished_at": finished,
                "duration_s": round(finished - self._current_started, 3),
            }
            self._recent.insert(0, entry)
            self._recent = self._recent[:MAX_RECENT]
            self._current_operation = None
            self._current_started = None
            self._current_progress = None
            self._write_state()

    def set_progress(self, current: int, total: int, detail: str = ""):
        """Update progress within a report() block."""
        self._current_progress = {
            "current": current,
            "total": total,
            "detail": detail,
        }
        self._write_state()

    def _write_state(self):
        if not self._active:
            return
        state = {
            "server": self._server_name,
            "pid": self._pid,
            "state": "busy" if self._current_operation else "idle",
            "operation": self._current_operation,
            "started_at": self._current_started,
            "progress": self._current_progress,
            "recent": self._recent,
            "updated_at": time.time(),
        }
        tmp = self._file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(state))
            tmp.replace(self._file)
        except OSError:
            logger.debug("Failed to write activity file: %s", self._file)

    def cleanup(self):
        """Remove the activity file on server shutdown."""
        if not self._active:
            return
        try:
            self._file.unlink(missing_ok=True)
        except OSError:
            pass
