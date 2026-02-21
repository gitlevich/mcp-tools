# Capture Conversation

Triggered by: "capture conversation", "capture conversations", "capture"

Export conversation transcripts and re-index the project embedding so previous conversations become searchable.

## Step 1: Export

Run the export conversations script:

```bash
./scripts/export_conversations.sh
```

## Step 2: Re-index

Trigger a full re-index of the project embedding via the `project-embed` MCP tool (`reindex`). This picks up the new or updated transcripts in `docs/genesis/`.

## Step 3: Verify

Search for a term from the current conversation using the `project-embed` MCP tool (`search`) to confirm the new content is indexed.

## Step 4: Stage and Commit

Stage the new or updated transcripts in `docs/genesis/` along with `.export_state.json`. Commit with a message like "Capture conversation transcripts".
