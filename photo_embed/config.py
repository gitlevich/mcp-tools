import os
from pathlib import Path

SERVICE_PORT = int(os.environ.get("PHOTO_EMBED_PORT", "7820"))
SERVICE_HOST = "127.0.0.1"
SERVICE_PID_FILE = Path("/tmp/mcp-tools/photo-embed.pid")
SERVICE_STARTUP_TIMEOUT = 60  # seconds to wait for health check

DEFAULT_TOP_K = 10
