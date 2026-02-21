"""Lightweight MCP client that spawns a server subprocess and calls tools via JSON-RPC over stdio."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class McpClient:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    _proc: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _req_id: int = field(default=0, init=False, repr=False)

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        await self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "dashboard", "version": "0.1.0"},
        })
        await self._notify("notifications/initialized")

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = await self._send("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        return result

    async def close(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.stdin.close()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()

    async def _send(self, method: str, params: dict) -> Any:
        self._req_id += 1
        msg = {"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params}
        return await self._write_and_read(msg)

    async def _notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

    async def _write_and_read(self, msg: dict) -> Any:
        line = json.dumps(msg) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

        expected_id = msg["id"]
        while True:
            raw = await asyncio.wait_for(self._proc.stdout.readline(), timeout=30)
            if not raw:
                raise ConnectionError("Server closed stdout")
            try:
                resp = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if resp.get("id") == expected_id:
                if "error" in resp:
                    raise RuntimeError(resp["error"])
                return resp.get("result")
