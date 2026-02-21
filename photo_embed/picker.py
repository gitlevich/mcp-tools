"""Native macOS Finder folder picker via osascript."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def pick_folder(prompt: str = "Select a photo folder") -> str | None:
    """Open a native macOS Finder dialog to pick a folder.

    Uses osascript (AppleScript) which has proper GUI access even when
    called from a background daemon process.
    """
    script = f'POSIX path of (choose folder with prompt "{prompt}")'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        path = result.stdout.strip().rstrip("/")
        if result.returncode != 0:
            # User cancelled or dialog failed
            logger.info("Folder picker cancelled or failed: %s", result.stderr.strip())
            return None
        return path if path else None
    except subprocess.TimeoutExpired:
        logger.warning("Folder picker timed out")
        return None
    except Exception:
        logger.warning("Folder picker failed", exc_info=True)
        return None
