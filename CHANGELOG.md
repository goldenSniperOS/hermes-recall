# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-30

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
  session summary, timeline) replacing the ~19 Engram MCP tool schemas sent on
  every API call.
- Per-profile state holding the last memory id, plus a `hermes recall status`
  CLI surface for it.
- Settings under `memory.recall.*`: `url`, `project`, `recall_limit`,
  `recall_max_chars` and `enabled`.
