---
name: remote-experiment-bridge
description: Use this skill when handing off long-running remote experiments between Codex sessions through a shared bridge directory, compact rolling reports, and token-saving notifications.
---

# Remote Experiment Bridge

Use this workflow when long-running experiments need to be monitored by one Codex session and handed off to another through a shared coordination server.

## Core Pattern

1. Create a shared bridge directory on a server reachable by both Codex sessions.
2. Write a handoff file with goals, exact commands, latest status, failure rules, and notification criteria.
3. Maintain a compact rolling report file for routine status.
4. Store detailed logs or snapshots in a separate subdirectory only when needed.
5. Notify in chat only for meaningful changes; otherwise update the report file silently.

## Recommended Bridge Layout

```bash
<bridge_root>/
  handoff_<project>_<date>.md
  experiment_monitor_report.md
  snapshots/
```

Use stable file names so a user can monitor progress without reopening long chat history.

## Required Handoff Content

The handoff should include:

- active goal and stop condition;
- experiment project root;
- bridge file paths;
- primary status command;
- latest known job IDs, nodes, checkpoints, and output paths;
- pending tasks and expected output files;
- known status-script bugs or metric-reading caveats;
- rules for resubmission, especially "do not blindly resubmit";
- notification criteria;
- token-saving policy.

## Rolling Report Format

```markdown
# Compact Experiment Monitor Report

Last updated: YYYY-MM-DD HH:MM timezone

## Current Snapshot

- Task A: state, latest checkpoint/output, next trigger.
- Task B: state, latest checkpoint/output, next trigger.

## Meaningful Changes Since Last Report

- None, or one bullet per real change.

## Next Check

- Suggested next check time or condition.
```

## Notification Rules

Notify the user only when one of these occurs:

- a training job starts, ends, fails, or is resubmitted;
- a new checkpoint or output changes downstream readiness;
- an evaluation starts, completes, or misses expected outputs;
- a blocked state persists and needs a decision;
- all expected outputs for a task are complete.

If there is no meaningful change, update only the rolling report with one timestamped sentence.

## Safety Rules

- Do not modify unrelated experiments.
- Do not blindly resubmit failed jobs. Inspect scheduler output and logs first.
- Do not paste long logs into chat; summarize them and store details in `snapshots/`.
- If the bridge host cannot reach the experiment host directly, use the receiving environment's existing SSH path while keeping reports in the bridge directory.
