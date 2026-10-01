# Feature: hermes-recall (Engram-compatible memory provider for Hermes)

## Objective

Ship `hermes-recall`: a Hermes Agent **memory provider** plugin that makes Engram the
agent's memory with the same always-on behavior gentle-engram gives Pi: recall before
every turn, capture after every turn, session lifecycle, compaction recovery, a visible
memory indicator, and the last memory id always at hand.

## Problem

- Hermes uses Engram today only as an MCP server: the model decides whether to search.
  It often starts coding without any recall.
- The MCP route sends ~19 Engram tool schemas on every API call (token cost).
- Telegram has no side bar, so the user cannot see what memory did or the last id (#761).

## Why

The user resumes work by memory id and needs recall/capture to be deterministic, not a
prompt request. Hermes ships a native `MemoryProvider` ABC for exactly this.

## Scope

- Exclusive memory-provider plugin, activated by `memory.provider: recall`.
- Talks to the local Engram HTTP server (`engram serve`, default http://127.0.0.1:7437),
  same endpoints gentle-engram uses. Same DB and ids as every other agent.
- Hooks: `system_prompt_block`, `prefetch` (+ `recall_status` for the 🧠 line),
  `queue_prefetch`, `on_turn_start` (prompt capture), `sync_turn` (passive capture),
  `on_pre_compress`, `on_session_switch`, `on_session_end`, `on_delegation`.
- Compact tool set replacing the MCP surface (mem_search, mem_save, mem_update,
  mem_get_observation, mem_context, mem_session_summary, mem_timeline).
- Visible status: 🧠 recall line (Hermes native) and 🧠 saved #id line on writes.
- Last memory id persisted per profile and exposed for hermes-odd (`/odd_status`).
- `hermes recall status` CLI + `/recall [id]` detail command if the surface allows.
- Docs: README with credits to Alan Buscaglia / Gentleman Programming, Built with
  Gentle AI badge, Not affiliated section, THIRD_PARTY_NOTICES, agent install prompt.

## Non-goals

- Starting/owning the Engram server process lifecycle beyond a health check.
- Engram Cloud auth flows.
- Any change to Hermes core.

## Constraints

- Python stdlib only (urllib); tests with unittest/pytest using a fake HTTP server.
- `sync_turn`/capture non-blocking (daemon worker); prefetch bounded (timeout, caps).
- Fail-soft: Engram down => no recall, no crash, one clear status.
- Name must not imply official Engram status (trademark policy): "Engram-compatible".
- Artifacts in English. No AI attribution in commits.

## Tasks

- [x] T1 Scaffold: plugin.yaml (kind exclusive), root `__init__.py` + package, pyproject,
      CI/release workflows, LICENSE, version-sync tests. Route: delegated (OpenCode).
- [x] T2 Engram HTTP client + project detection (`GET /project/current?cwd=`), health,
      timeouts, fail-soft. Route: delegated.
- [x] T3 Provider core: availability, initialize (session register), system prompt
      block, prefetch with caps + recall_status, queue_prefetch. Route: delegated.
- [x] T4 Capture: on_turn_start prompt capture, sync_turn passive capture (tool results
      + assistant text), on_delegation. Background worker. Route: delegated.
- [x] T5 Lifecycle: on_session_switch, on_session_end, on_pre_compress (archive summary,
      recovery context). Route: delegated.
- [x] T6 Tools: compact mem_* schemas + handle_tool_call; 🧠 saved #id status; last id
      state file. Route: delegated.
- [ ] T7 Docs + credits + install prompt; real install in throwaway HERMES_HOME; v0.1.0
      release; public repo. Route: inline.
- [ ] T8 hermes-odd v0.7.0: doctor detects provider, status shows last memory id,
      install prompt covers hermes-recall, memory pointer adapts. Route: delegated.

## Acceptance criteria

- With `memory.provider: recall` and Engram running, every non-trivial turn shows
  `🧠 Recall — recalled N memories` (when hits) without the model calling a tool.
- Each completed turn posts prompt + passive capture to Engram under the detected project.
- A memory saved from Hermes is visible from Pi/OpenCode with the same id, and vice versa.
- Engram down: Hermes keeps working; status reports offline.
- Tests green on 3.11/3.12; `hermes plugins install` works in a throwaway home.

## Checks

- `python3.11 -m pytest -q` (or Hermes venv), ruff, real install smoke.

## TDD

Mode: standard test-first where a runnable test and clear expected outcome exist
(upstream ODD default). Runner: pytest.

## Delivery

Strategy: single-pr not used; first version straight to `main` (new repo).
Forecast: ~1200 authored lines across T1-T6 (over the 400 heuristic; explained: new repo).

## Upstream references

- gentle-engram (Engram repo `plugin/pi`) at a65917f3072ab4f26780bcf52427eba3963bef46
- Engram HTTP server `internal/server/server.go` same commit.

## Progress

- T1 scaffold: `15e93a0` (OpenCode writer).
- T2 Engram client: `2e98c45` (OpenCode writer).
- T3 provider core: `727d6d0` (OpenCode writer).
- T4 capture: `85ebc54` (OpenCode writer).
- T5 lifecycle: `6b3e8b8` (OpenCode WIP finished by a Pi gentle-ai-worker after the
  OpenCode run was cancelled by the owner).
- T6 tools, state file, /recall, CLI, design doc: `9482d0a` (Pi gentle-ai-worker).
  Scope decision: 10 tools (the 7 planned + mem_judge, mem_pin, mem_unpin).
- T7 docs: `47a710c` (inline). Throwaway HERMES_HOME install from the local repo:
  `memory status` shows `Provider: recall ... available`, a real `hermes chat -q`
  turn showed `🧠 Recall · hermes-recall — recalled 3 memories`, prompt captured,
  session registered and ended, `state.json` written.
- Fix pass commit: `96ffb77`, approved by native review (lineage
  review-625857e7947d058e, 4 lenses, 8 non-blocking advisories).
- v0.1.0 tagged at `96ffb77`, public repo goldenSniperOS/hermes-recall, CI and
  release green (4 assets). GitHub install was then blocked by the Hermes install
  scanner (`sudo_usage` in the install prompt); fixed in v0.1.1 with a scanner
  regression test.
- Independent verification (gentle-ai-verify): no blockers; 2 majors (private-tag
  redaction after truncation in compaction/delegation, no session registration
  retry) and minors. Fix pass (Pi gentle-ai-worker, test-first): redaction before
  truncation, session registration retry, queue_prefetch reduced to project
  warm-up, no recapture after a session switch, state.json only from the primary
  context, loopback without proxies, MCP Engram results not captured, real
  Hermes loader test. 593 tests passed, ruff clean.

## Next step

Release v0.1.1, re-check the GitHub install in a throwaway home, then the live
profile install (owner approval for memory.provider). Follow-ups for v0.2:
review advisories (pre-registration doubling outage cost, delegation non-str
guard, pip-install CLI gap). T8 is owned by the hermes-odd session.
