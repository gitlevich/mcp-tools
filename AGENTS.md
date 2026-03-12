# Project Instructions

## Sigils

When the user invokes a sigil by name, read its definition and follow it.

- "export conversations" -> `docs/sigils/export_conversations.md`
- "capture conversation" / "capture" -> `docs/sigils/capture_conversation.md`
- "search genesis" -> `docs/sigils/search_genesis.md`

## Tools

### rename

AST-aware symbol rename across Python, TypeScript, and Swift. One call renames a symbol and updates all references across the project.

```bash
uv run --directory "/Users/vlad/Attention Lab/mcp-tools/rename" python rename_cli.py <file_path> <old_name> <new_name>
```

Outputs JSON with `old_name`, `new_name`, and `files_changed`. On error outputs `{"error": "..."}`.

Swift rename uses SourceKit-LSP. For Xcode projects, a recent build in Xcode is needed so the index is fresh.

## Services

### photo-embed

The photo-embed MCP server (in `photo_embed/`) is a thin client. The service lives in a separate project at `/Users/vlad/Attention Lab/sense-maker/`.

To (re-)start the service after code changes:

```bash
"/Users/vlad/Attention Lab/sense-maker/up.sh"
```

This kills any running instance, then launches `service.py` via `uv run` in sense-maker's `.venv`. Logs go to `/tmp/mcp-tools/photo-embed.log`.
