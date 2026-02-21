# Genesis

Conversation transcripts between the user and Claude Code, exported as readable markdown.

Each file records one session: who said what, when, and which project files were touched. File references are relative links, clickable on GitHub.

## How to export

Run the export script:

```bash
./scripts/export_conversations.sh
```

This reads Claude Code's session JSONL from `~/.claude/projects/` and appends new turns to the transcripts here. It is idempotent — re-running it only adds what's new.

Deduplication state is tracked in `.export_state.json`.
