# Prompt log — ein

Append-only. Never overwrite or truncate previous entries.

## [2026-09-23 23:49 IST]

Task:
- Set up a persistent per-owner prompt + contribution audit trail.

Prompt summary:
- Auto-detect owner from path/branch (~/amlc26-harness or `main` → ein; agy1/agy2 worktrees → agy1/agy2).
- Maintain `docs/prompts/<owner>.md` and `docs/contributions/<owner>.md`; read both at session start, append only, never rewrite history.
- Log every meaningful task (prompt summary, expected vs actual result, exp id, commit) and every completed task (files changed, CV/results, commit, decision, next).
- Only own-owner files; no secrets or raw data; follow AGENTS.md; no pointless commits; end each task with a 3-line report.

Expected output:
- Both log files created for owner `ein` and the setup recorded.

Actual result:
- Created `docs/prompts/ein.md` and `docs/contributions/ein.md` with this entry. No code changed; not committed (logs will be committed with the next real change).

Related experiment:
- None.

Related commit:
- None.

## [2026-09-23 23:59:07+05:30]

Task:
- Establish manual handoff and routing rules for inter-agent/team communication.

Prompt summary:
- Do not automatically communicate with AGY2, TM1, or TM2.
- Human operator manually copies outputs between sessions.
- Final response must always include a specific 6-part structure:
  1. WHAT CHANGED
  2. RESULT / CV
  3. NEXT DECISION
  4. DESTINATION (T0, AGY1, AGY2, TM1, TM2, or "No handoff required")
  5. EXACT PROMPT TO PASTE
  6. RETURN ROUTING
- Provide complete ready-to-paste prompts for destination actors and explicit instructions on where results should be returned.

Expected output:
- Acknowledge instructions, update logs, and adopt the new 6-part reporting structure for all future responses.

Actual result:
- Logged new instructions. Adopting the 6-part reporting structure.

Related experiment:
- N/A

Related commit:
- N/A
