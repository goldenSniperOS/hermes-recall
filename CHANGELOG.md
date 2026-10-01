# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
