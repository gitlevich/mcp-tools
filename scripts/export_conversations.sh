#!/usr/bin/env bash
# Export Claude Code conversations to docs/genesis/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/export_conversations.py"
