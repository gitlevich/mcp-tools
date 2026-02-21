"""macOS Finder folder picker via AppleScript."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def pick_folder(prompt: str = "Select a folder to index") -> str | None:
    """Open a native macOS Finder dialog to pick a folder.

    Uses osascript (AppleScript) which reliably shows a folder picker
    from any process context. Returns the selected path or None if cancelled.
    """
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                f'POSIX path of (choose folder with prompt "{prompt}")',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        path = result.stdout.strip()
        if result.returncode != 0:
            # returncode 1 = user cancelled
            if result.returncode == 1:
                return None
            logger.warning("Picker process failed: %s", result.stderr)
            return None
        return path if path else None
    except subprocess.TimeoutExpired:
        logger.warning("Folder picker timed out")
        return None
    except Exception:
        logger.warning("Folder picker failed", exc_info=True)
        return None
