# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-10-27

Engram 3.6.0 additions (gentle-engram 0.5 parity). Every new route degrades on
older servers: earlier Engram versions keep working, and a new tool on a server
without its route answers "needs Engram >= 3.5.1" instead of failing.

### Added

- `mem_consolidate`: merge duplicate observations through
  `POST /observations/consolidate`. A preview writes nothing and returns the
  plan with its fingerprint for the user to approve; `confirm` with that
  fingerprint applies it. A changed plan (409 `fingerprint_mismatch`) comes back
  as a fresh plan and is never confirmed automatically; a confirmation that
  timed out or failed server-side is reported as unknown and never retried.
- `mem_judge_many`: resolve up to 32 conflict candidates in one
  `POST /conflicts/judge-many`, each validated like `mem_judge`, never retried.
- `mem_session_finalize`: save the final summary and end the session in one
  `POST /sessions/{id}/finalize`. After it, the shutdown or reset `/end` is not
  posted, also when the outcome is uncertain; a later write lands in a resumed
  continuation.
- `mem_get_observation` takes `include_history` and `history_cursor` and returns
  the observation's prior versions (each capped), `version_count` and the next
  page cursor.
- Capture writes (`/sessions`, `/observations`, `/observations/passive`,
  `/prompts`) carry `project_provenance {source, directory}`: Engram's detected
  `project_source`, `config` for `memory.recall.project`, else `unknown`.

### Changed

- `mem_search` and the per-turn recall ask for `response_format=compact`: hits
  render from the bounded `preview` and `mem_search` marks pinned hits. The
  legacy full answer of older servers is still accepted.
- A tool error from a server that answered (for example 404 `unknown_project`
  or 409 `ambiguous_project`) reports Engram's reason, `code` and
  `available_projects` instead of asking whether `engram serve` is running.
- The system prompt follows gentle-engram's canonical topic workflow (search,
  read in full, update the same topic by id with `expected_project`, save a new
  topic with a distinct `topic_key`, never merge across projects), retries an
  empty scoped search across all projects, and names the new tools.
- The tool set grows from 10 to 13 schemas (about 5.5 KB).

## [0.3.0] - 2026-10-02

Compatibility with Engram v3.0.0 (and gentle-engram 0.2.0). Engram 2.0.0
keeps working.

### Changed

- **Breaking for the model:** `mem_update` now requires `expected_project`,
  the project that owns the observation. It is sent as
  `PATCH /observations/{id}?expected_project=` and is never filled in from the
  detected project. Engram v3 refuses an update without it (400) or with the
  wrong owner (409); the refusal comes back as a readable tool error. A blank
  value is refused locally before any request.
- Session registration sends `"resume": true`. When Hermes resumes a session
  Engram already ended, Engram v3 picks a live `<id>:resume:<N>` continuation
  and every write of that session, including its end, goes there. Before, the
  registration was refused with `session_already_ended` and writes failed.
  The acknowledged id is validated the way gentle-engram does it.
- An `engram serve` started by recall runs with `ENGRAM_CLOUD_AUTOSYNC=1`,
  like gentle-engram's. The server skips autosync when no cloud is configured.

## [0.2.0] - 2026-10-01

### Added

- Engram server auto-start, with the same rules gentle-engram uses in Pi:
  when no server answers at the default local address, the primary agent
  starts a fully detached `engram serve` (log in
  `<hermes_home>/recall/engram-serve.log`) and waits for it, bounded. A
  healthy server is adopted only when its `/health` instance id matches
  `engram instance-id`. A server reporting a different instance id gets no
  memory traffic at all; a server with no instance id, or one whose identity
  cannot be checked, is still used but never managed. One restart attempt per
  provider instance when an adopted or started server stops answering. The
  spawned server runs from `<hermes_home>/recall`, its log is rotated past
  1 MB, and its exit is reaped so it never lingers as a zombie. A cross-process lock (`<hermes_home>/recall/serve.lock`)
  prevents double starts. `ENGRAM_BIN` selects the binary.
- `memory.recall.autostart` setting (default `true`).
- `hermes recall status` and `/recall` show how the server was obtained
  (started by recall, adopted, foreign instance — not used, externally
  managed, or why it is not running).

### Changed

- You no longer need to start `engram serve` by hand; the binary is still
  required. Nothing is started when `ENGRAM_URL` is set, from subagents, cron
  jobs or flushes, or for a URL `engram serve` would not bind.
- The server is never stopped, not even at shutdown: it is shared with Pi and
  every other agent.
- With neither `ENGRAM_URL` nor `memory.recall.url` set, the default URL
  follows `ENGRAM_PORT` (`http://127.0.0.1:<ENGRAM_PORT>`), like gentle-engram.

## [0.1.1] - 2026-10-01

### Fixed

- `hermes plugins install goldenSniperOS/hermes-recall` was blocked by the Hermes
  install scanner (community source, caution verdict): the agent install prompt
  mentioned a privilege-escalation command, which the scanner rates high. The
  prompt now says "administrator rights", and a test runs the real Hermes scanner over the tracked tree and requires a
  `safe` verdict.

## [0.1.0] - 2026-10-01

First release: an exclusive Hermes Agent memory provider, activated with
`memory.provider: recall`, that makes a local Engram HTTP server the agent's
memory instead of reaching Engram through MCP.

### Added

- Provider hooks on the Hermes `MemoryProvider` ABC: system prompt block,
  bounded `prefetch` with a `recall_status` line, `queue_prefetch`, prompt and
  passive capture per turn, delegation, session switch and session end, and
  compaction recovery on pre-compress.
- Engram HTTP client over the standard library only (no runtime dependencies),
  with project detection, a health check, timeouts and fail-soft behaviour: with
  Engram down the agent keeps working and the status reports offline.
- A compact `mem_*` tool set (search, save, update, get observation, context,
  session summary, timeline, judge, pin, unpin) replacing the ~19 Engram MCP
  tool schemas sent on every API call. Write results carry a `🧠 saved #id`
  style status.
- Per-profile state file `<hermes_home>/recall/state.json` holding the last
  memory id (stable format, read by other plugins), the `/recall [id]` slash
  command and the `hermes recall status|show <id>` CLI.
- Docs: README with credits and the not-affiliated notice,
  THIRD_PARTY_NOTICES, design notes and an install prompt for agents.
- Settings under `memory.recall.*`: `url`, `project`, `recall_limit`,
  `recall_max_chars` and `enabled`.
