"""Native macOS Finder folder picker via NSOpenPanel."""

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def pick_folder(prompt: str = "Select a photo folder") -> str | None:
    """Open a native macOS Finder dialog to pick a folder.

    Runs the dialog in a subprocess to avoid AppKit main-thread issues
    in the MCP server process. Returns the selected path or None if cancelled.
    """
    script = f"""
import AppKit
import objc

panel = AppKit.NSOpenPanel.openPanel()
panel.setCanChooseDirectories_(True)
panel.setCanChooseFiles_(False)
panel.setAllowsMultipleSelection_(False)
panel.setMessage_("{prompt}")
panel.setPrompt_("Select")

# Bring the panel to front
AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

result = panel.runModal()
if result == AppKit.NSModalResponseOK:
    print(panel.URL().path(), end="")
else:
    print("", end="")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        path = result.stdout.strip()
        if result.returncode != 0:
            logger.warning("Picker process failed: %s", result.stderr)
            return None
        return path if path else None
    except subprocess.TimeoutExpired:
        logger.warning("Folder picker timed out")
        return None
    except Exception:
        logger.warning("Folder picker failed", exc_info=True)
        return None
