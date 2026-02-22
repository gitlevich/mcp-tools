# Project Instructions

## Sigils

When the user invokes a sigil by name, read its definition and follow it.

- "export conversations" -> `docs/sigils/export_conversations.md`
- "capture conversation" / "capture" -> `docs/sigils/capture_conversation.md`
- "search genesis" -> `docs/sigils/search_genesis.md`

## Services

### photo-embed

To (re-)start after code changes:

```bash
photo_embed/up.sh
```

This kills any running instance, then launches `service.py` via `uv run` in the project's `.venv`. Logs go to `/tmp/mcp-tools/photo-embed.log`. Must use `uv run` — the project venv is managed by uv, not the system Python.
