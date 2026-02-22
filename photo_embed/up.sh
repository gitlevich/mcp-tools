#!/usr/bin/env bash
# (Re-)start the photo-embed service.
# Kills any existing instance, then launches a new one via uv run.

set -euo pipefail
cd "$(dirname "$0")"

PID_FILE="/tmp/mcp-tools/photo-embed.pid"

if [[ -f "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        kill "$old_pid"
        echo "Killed old service (PID $old_pid)"
    fi
    rm -f "$PID_FILE"
fi

nohup uv run python service.py > /tmp/mcp-tools/photo-embed.log 2>&1 &

# Wait for PID file to appear
for i in $(seq 1 10); do
    if [[ -f "$PID_FILE" ]]; then
        echo "Service started (PID $(cat "$PID_FILE"))"
        exit 0
    fi
    sleep 1
done

echo "Service failed to start. Check /tmp/mcp-tools/photo-embed.log"
exit 1
