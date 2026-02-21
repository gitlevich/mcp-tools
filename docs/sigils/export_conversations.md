# Export Conversations

Triggered by: "export conversations", "save conversations", "export transcripts"

The user asks me to export Claude Code conversation transcripts to the project.

## Step 1: Run the Export

```bash
./scripts/export_conversations.sh
```

This reads all session JSONL files from `~/.claude/projects/` for this project and writes readable markdown transcripts to `docs/genesis/`. Each session becomes one file, named by date and session slug.

The script is append-only: existing transcripts gain new turns without overwriting. Deduplication state is tracked in `docs/genesis/.export_state.json`.

## Step 2: Review

Skim the output for session count and turn counts. If a session shows 0 new turns, it was already up to date.

## Step 3: Stage and Commit

Stage the new or updated transcripts in `docs/genesis/` along with `.export_state.json`. Commit with a message like "Capture conversation transcripts".
