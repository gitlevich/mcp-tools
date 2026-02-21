# Search Genesis

Triggered by: inability to find the answer to a question in the current conversation or project files.

When I cannot answer a question from what is in the current chat context or in the project's files, I search the conversation transcripts in `docs/genesis/`. These contain the full history of design conversations — decisions, rationale, rejected alternatives, and context that may not have made it into the specs.

## When to Enter

- The user asks about a past decision and I do not see it in the specs or awareness docs.
- I need context for why something was designed a certain way.
- A term or concept was discussed but I cannot find its definition in the current project files.

## How to Search

Use the semantic search tool (`mcp__project-embed__search`) to query `docs/genesis/` with a natural language description of what I am looking for. The transcripts are indexed and searchable by meaning, not just keyword.

If semantic search is unavailable, fall back to `Grep` across `docs/genesis/*.md`.

## What to Do with Results

Report what I found, citing the source transcript and the relevant exchange. Do not present genesis content as current truth — it is historical. The specs are truth. Genesis explains how they got there.
