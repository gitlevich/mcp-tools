"""Dashboard server: queries each MCP tool's runtime status and serves a simple web UI."""

import asyncio
import json
import logging
import os
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread

from mcp_client import McpClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SERVERS = {
    "project-embed": {
        "command": "/opt/homebrew/bin/uv",
        "args": ["run", "--directory", str(PROJECT_ROOT / "project_embed"), "python", "server.py"],
        "status_tools": [{"name": "index_status"}],
        "reindex_tool": "reindex",
    },
    "photo-embed": {
        "command": "/opt/homebrew/bin/uv",
        "args": ["run", "--directory", str(PROJECT_ROOT / "photo_embed"), "python", "server.py"],
        "status_tools": [{"name": "index_status"}],
        "reindex_tool": "reindex",
        "refresh_tool": "refresh",
    },
    "thought-index": {
        "command": "/opt/homebrew/bin/uv",
        "args": ["run", "--directory", str(PROJECT_ROOT / "thought_index"), "python", "server.py"],
        "status_tools": [{"name": "list_sources"}],
        "reindex_tool": "reindex",
    },
    "rename": {
        "command": "/opt/homebrew/bin/uv",
        "args": ["run", "--directory", str(PROJECT_ROOT / "rename"), "python", "server.py"],
        "status_tools": [],
        "reindex_tool": None,
    },
}


async def query_server(name: str, config: dict) -> dict:
    """Connect to one MCP server, call its status tools, return results."""
    result = {"name": name, "status": "unknown", "tools": {}, "error": None}
    client = McpClient(command=config["command"], args=config["args"])

    try:
        await client.start()
        result["status"] = "running"

        for tool in config["status_tools"]:
            try:
                resp = await client.call_tool(tool["name"], tool.get("args"))
                text = ""
                if isinstance(resp, dict) and "content" in resp:
                    for block in resp["content"]:
                        if block.get("type") == "text":
                            text += block["text"]
                elif isinstance(resp, str):
                    text = resp
                else:
                    text = json.dumps(resp)
                result["tools"][tool["name"]] = text
            except Exception as e:
                result["tools"][tool["name"]] = f"Error: {e}"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    finally:
        await client.close()

    return result


async def collect_all_status() -> list[dict]:
    """Query all servers concurrently."""
    tasks = [query_server(name, cfg) for name, cfg in SERVERS.items()]
    return await asyncio.gather(*tasks)


async def refresh_server(name: str) -> dict:
    """Spawn a server, call its refresh tool, return the result."""
    config = SERVERS.get(name)
    if not config or not config.get("refresh_tool"):
        return {"error": f"No refresh tool for '{name}'"}

    client = McpClient(command=config["command"], args=config["args"])
    try:
        await client.start()
        resp = await asyncio.wait_for(
            client.call_tool(config["refresh_tool"]),
            timeout=300,
        )
        text = ""
        if isinstance(resp, dict) and "content" in resp:
            for block in resp["content"]:
                if block.get("type") == "text":
                    text += block["text"]
        elif isinstance(resp, str):
            text = resp
        else:
            text = json.dumps(resp)
        return {"result": text}
    except Exception as e:
        return {"error": str(e)}
    finally:
        await client.close()


async def reindex_server(name: str) -> dict:
    """Spawn a server, call its reindex tool, return the result."""
    config = SERVERS.get(name)
    if not config or not config.get("reindex_tool"):
        return {"error": f"No reindex tool for '{name}'"}

    client = McpClient(command=config["command"], args=config["args"])
    try:
        await client.start()
        resp = await asyncio.wait_for(
            client.call_tool(config["reindex_tool"]),
            timeout=300,
        )
        text = ""
        if isinstance(resp, dict) and "content" in resp:
            for block in resp["content"]:
                if block.get("type") == "text":
                    text += block["text"]
        elif isinstance(resp, str):
            text = resp
        else:
            text = json.dumps(resp)
        return {"result": text}
    except Exception as e:
        return {"error": str(e)}
    finally:
        await client.close()


ACTIVITY_DIR = Path("/tmp/mcp-tools")
DASHBOARD_DIR = Path(__file__).resolve().parent


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/status":
            self._handle_status()
        elif self.path == "/api/activity":
            self._handle_activity()
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/refresh/"):
            server_name = self.path.split("/api/refresh/", 1)[1]
            self._handle_refresh(server_name)
        elif self.path.startswith("/api/reindex/"):
            server_name = self.path.split("/api/reindex/", 1)[1]
            self._handle_reindex(server_name)
        else:
            self.send_error(404)

    def _handle_refresh(self, server_name: str):
        try:
            result = asyncio.run(refresh_server(server_name))
            body = json.dumps(result).encode()
            status = 200 if "result" in result else 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    def _handle_reindex(self, server_name: str):
        try:
            result = asyncio.run(reindex_server(server_name))
            body = json.dumps(result).encode()
            status = 200 if "result" in result else 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    def _handle_activity(self):
        results = []
        if ACTIVITY_DIR.is_dir():
            for f in sorted(ACTIVITY_DIR.glob("*.json")):
                try:
                    data = json.loads(f.read_text())
                    pid = data.get("pid")
                    alive = False
                    if pid:
                        try:
                            os.kill(pid, 0)
                            alive = True
                        except OSError:
                            alive = False
                    data["alive"] = alive
                    results.append(data)
                except (json.JSONDecodeError, OSError):
                    continue

        body = json.dumps(results).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_status(self):
        try:
            results = asyncio.run(collect_all_status())
            body = json.dumps(results).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        logger.info(format, *args)


def main():
    port = int(os.environ.get("DASHBOARD_PORT", "8111"))
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    logger.info("Dashboard at http://127.0.0.1:%d", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
