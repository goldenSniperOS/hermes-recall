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

- Stopping or replacing an Engram server the provider did not verify (it adopts a
  healthy server with a matching instance id and never kills one).
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
- [x] T7 Docs + credits + install prompt; real install in throwaway HERMES_HOME; v0.1.0
      release; public repo. Route: inline.
- [x] T9 v0.2.0 Engram server auto-start, parity with gentle-engram: adopt a healthy
      server with matching instance id, spawn a detached `engram serve` when none
      answers (default loopback URL, ENGRAM_URL unset, primary context only), one
      bounded restart, cross-process lock, `memory.recall.autostart`, never kill.
      Route: delegated (Pi gentle-ai-worker), then verify + native review.
- [x] T10 Live install on the owner's Hermes (authorized): hermes-recall v0.2.0,
      `memory.provider recall`, keep the Engram MCP for now, hermes-odd v0.7.0;
      owner sends /restart. Route: inline.
- [x] T11 v0.3.0 Engram v3 compatibility (gentle-engram 0.2.0 / Engram v3.0.0):
      `mem_update` requires `expected_project` (PATCH `?expected_project=`, breaking in
      v3), session registration sends `resume: true` and adopts the server-selected
      effective id (`<id>:resume:N`), auto-started `engram serve` gets
      `ENGRAM_CLOUD_AUTOSYNC=1`. Must keep working on Engram 2.0.0.
      Route: delegated (Pi gentle-ai-worker), then verify + native review.
- [x] T8 (owned by the hermes-odd session) hermes-odd v0.7.0: doctor detects provider, status shows last memory id,
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

- T9 auto-start: `40e92ae` (Pi gentle-ai-worker, test-first; independent verify
  found 2 majors, fixed: foreign-instance gating and ENGRAM_PORT default URL).
  Real-binary smoke on an isolated port and data dir: `started` in 0.52 s, then
  `adopted`; the live server on 7437 classifies as `adopted`. Native review
  approved (review-e677f03752b50710, 7 non-blocking advisories). v0.2.0 tagged,
  CI and release green, GitHub install in a throwaway home passes the scanner.
  676 tests.
- T10 live install: the default profile already had recall 0.1.1 +
  `memory.provider recall` + hermes-odd 0.7.0 (installed 10:38, not by this
  session); updated recall to 0.2.0 (`40e92ae`). Engram MCP kept. Profile
  `buscador-casa` untouched. Waiting for the owner's /restart.

- T11 Engram v3 compatibility (Pi gentle-ai-worker, test-first): mem_update requires
  `expected_project`, session registration sends `resume: true` and writes go to the
  validated effective id, spawned `engram serve` gets `ENGRAM_CLOUD_AUTOSYNC=1`.
  Independent verify: PASS, 4 minors fixed (echoing fixtures, no-id ack rejected like
  upstream, `last_error_status` instead of substring matching, docs). Real smoke on
  isolated ports against Engram 2.0.0 and 3.0.0: resume yields `:resume:2` and stays
  stable on v3, wrong owner is a 409 on v3, blank owner refused locally. Native review
  approved (review-e6dd28a6e37b6772, 4 non-blocking advisories: second idempotent /end
  on 2.0.0 after resume, autosync overrides a user `ENGRAM_CLOUD_AUTOSYNC=0` like
  upstream, effective-id map bound, tools.py:525). 723 tests.

## Next step

Owner sends /restart and checks the 🧠 line and /odd_status. v0.2.x follow-ups:
review advisories (server.py docstrings, "unavailable" overloaded, tools.py:619),
pre-registration outage cost, delegation non-str guard, pip-install CLI gap;
optionally remove the Engram MCP once recall is field-proven.
