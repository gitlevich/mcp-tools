# Session: graceful-booping-shannon
**Date**: 2026-02-21 06:01 UTC  
**Session ID**: `ca73fcaa-1e4f-41c5-877d-cee204fd8169`

---

### User (06:01 UTC)

Educate me about teams in Claude Code.

### Assistant (06:02 UTC)

## Teams in Claude Code

There are two separate concepts:

---

### 1. Organizational Plans (Teams / Enterprise)

These are billing and admin tiers for using Claude Code within an organization:

- **Claude for Teams** — self-service plan with centralized billing, admin tools, and collaboration features. Good for smaller teams.
- **Claude for Enterprise** — adds SSO, domain capture, role-based permissions, compliance API, and managed policy settings.

Members authenticate via their Claude.ai accounts. Admins get a dashboard for billing, device management, and settings.

---

### 2. Agent Teams (Experimental)

This is a parallel multi-agent feature where multiple Claude Code instances coordinate on a single task:

**How it works:**
- One session acts as the **team lead**, spawning and managing **teammates**
- Each teammate runs independently with its own context window
- Teammates can **message each other directly** (unlike subagents, which only report back to the parent)
- They share a task list with automatic dependency tracking

**Enabling it:**
```
CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```
Set this in environment or `settings.json`.

**Key differences from subagents:**
| | Subagents | Agent Teams |
|---|---|---|
| Communication | Only back to parent | Peer-to-peer between teammates |
| Token usage | Lower | ~7x more |
| Best for | Quick, focused tasks | Complex work needing coordination |

**Good use cases:**
- Parallel code reviews (security, performance, tests)
- Multi-module feature development (frontend, backend, tests simultaneously)
- Research with competing hypotheses

**Limitations:**
- Experimental and disabled by default
- One team per session, no nested teams
- Requires tmux or iTerm2 for split-pane mode
- No session resumption with in-process teammates
- Significantly higher token costs (~7x)

**Practical tips:**
- Use Sonnet for teammates to manage costs
- Keep teams small
- Clean up teammates when done

---

Want me to dig deeper into any specific aspect?

### User (06:04 UTC)

Please enable this feature.

### Assistant (06:05 UTC)

Done. Agent teams is now enabled globally in [settings.json](/Users/vlad/.claude/settings.json). The change will take effect in new Claude Code sessions.
