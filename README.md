# mcp-tools

Two MCP (Model Context Protocol) servers for use with Claude Code and Cursor: semantic code search and AST-aware symbol renaming.

## Servers

### project-embed

Semantic code search powered by OpenAI embeddings and ChromaDB. Indexes your project's `.py`, `.ts`, `.tsx`, and `.md` files into a local vector database, then answers natural language queries by returning the most relevant code chunks.

**Tools exposed:**
- `search_code(query, top_k=10)` — find code by meaning, not keywords
- `index_status()` — check indexing state
- `reindex()` — force full rebuild

The index refreshes incrementally on each search (mtime-based), so it stays current without manual rebuilds.

**Requirements:** OpenAI API key in environment (`OPENAI_API_KEY`).

### rename

AST-aware symbol renaming across an entire codebase. Renames functions, classes, variables, interfaces, and type aliases — updating all references, including imports.

- **Python**: uses [Rope](https://github.com/python-rope/rope)
- **TypeScript**: uses [ts-morph](https://github.com/dsherret/ts-morph)

**Tool exposed:**
- `rename_symbol(file_path, old_name, new_name)` — rename and update all references

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Node.js (for TypeScript renaming).

```bash
# Install Python dependencies
cd project_embed && uv venv && uv pip install -e ".[dev]"
cd ../rename && uv venv && uv pip install -e ".[dev]" && npm install
```

## Configuration

Add to your project's `.mcp.json` (Claude Code) or `~/.cursor/mcp.json` (Cursor):

```json
{
  "mcpServers": {
    "project-embed": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-tools/project_embed", "python", "server.py"],
      "env": {
        "PROJECT_ROOT": "/path/to/your/project"
      }
    },
    "rename": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-tools/rename", "python", "server.py"],
      "env": {
        "PROJECT_ROOT": "/path/to/your/project"
      }
    }
  }
}
```

### Environment variables

| Variable | Server | Default | Description |
|---|---|---|---|
| `PROJECT_ROOT` | both | cwd | Root of the project to index/rename in |
| `CHROMA_DIR` | project-embed | `$PROJECT_ROOT/.cache/project-embed` | Where to store the vector index |
| `TSCONFIG_PATH` | rename | `$PROJECT_ROOT/frontend/tsconfig.json` | Path to tsconfig for TypeScript projects |

## Tests

```bash
cd project_embed && uv run pytest tests/ -v
cd ../rename && uv run pytest tests/ -v
```
