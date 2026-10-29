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

## v0.4.0 — Engram 3.1-3.6 / gentle-engram 0.5 additions

Target: Engram 3.6.0 (installed via brew 2026-10-27; DB backup
`~/.engram-backups/engram-2026-10-27-pre-3.6.db`). Contract mapped from
`.refs/engram-src` at v3.6.0 (98912f8). v0.3.0 already passes a full smoke on 3.6.0.
New routes have no `/health` capability flag: an older server's 404/405 must turn into
a clear "needs Engram >= 3.5.1" tool error, never a crash.

### Specs

- S1 Compact search: `mem_search` and the prefetch `/search` send
  `response_format=compact`; the client accepts both the compact object
  `{result, results:[{id, title, type, preview, truncated, pinned, state, ...}]}` and
  the legacy bare array (older servers ignore the param). Hits render from `preview`
  when there is no `content`; `pinned` is surfaced in tool hits.
- S2 History: `mem_get_observation{id, include_history?, history_cursor?}` forwards
  `include_history=true` and `history_cursor`; returns `history`, `version_count`,
  `history_truncated`, `history_cursor` (next page); each history entry's content is
  capped. Without `include_history` the request is unchanged.
- S3 `mem_consolidate{ids_to_delete[>=2 distinct positive ints], new_title,
  new_content, expected_project, type?, topic_key?, confirm?, fingerprint?}` → POST
  `/observations/consolidate`. Preview (no confirm) returns the plan and tells the
  model to show it to the user and confirm with the plan's fingerprint. A 409
  `fingerprint_mismatch` returns the fresh plan and is never auto-confirmed; no
  retry after a 500 or timeout. Write tool (primary context only); private tags
  redacted from title/content; survivor id remembered.
- S4 `mem_judge_many{judgments[1..32]{judgment_id, relation, reason?, evidence?,
  confidence?}}` → POST `/conflicts/judge-many` with the effective session id;
  per-item validation like `mem_judge`; outcomes returned; never replayed.
- S5 `mem_session_finalize{summary, project?}` → POST `/sessions/{effective id}/finalize`
  `{project, summary}` (registers the session first, like saves). Success (incl.
  `already_finalized`) marks the session ended so the shutdown/switch `/end` is not
  posted, and a later write registers a resume continuation. Uncertain result: no
  retry, no fallback to `/end`. Refusals (403/404/409) are reported with Engram's code.
- S6 Capture provenance: `project_provenance {source, directory}` on POST `/sessions`,
  `/observations`, `/observations/passive`, `/prompts`. `source` is the detected
  `project_source` when the detected project is the write's project and the value is
  in Engram's allowed set, `explicit_override` for `memory.recall.project` (L11), else
  `unknown`.
  Never sent on `/finalize` (strict body). `/project/current` has no `directory` param
  in 3.6.0 (`cwd` is canonical, already used): not applicable.
- S7 Protocol: the static system prompt adds the canonical topic workflow (search
  first, read in full, same topic → `mem_update` by id with `expected_project`,
  new topic → `mem_save` with a distinct `topic_key`, never merge across projects),
  the empty-scoped-search retry (`match_mode:any` + `all_projects`), and the new
  tools. Compaction recovery already travels outside the system prompt
  (`on_pre_compress` return value): documented, no change.
- S8 Server refusals are not outages: a read tool that got an HTTP answer (e.g. 404
  `unknown_project` with `available_projects`, 409 `ambiguous_project`) reports
  Engram's reason instead of "Is `engram serve` running?".

### Tasks

- [x] T12 S2+S4+S8 client error bodies, history, judge-many, refusal wording. Route: inline.
- [x] T13 S3 mem_consolidate. Route: inline.
- [x] T14 S5 mem_session_finalize + end suppression. Route: inline.
- [x] T15 S1 compact search (tool + prefetch). Route: inline.
- [x] T16 S6 capture provenance. Route: inline (own commit; writes data).
- [x] T17 S7 protocol text, README/design/CHANGELOG, v0.4.0 version bump. Route: inline.
- [x] T18 Verify (real Engram 3.6.0 isolated smoke), native review per RDD, release
      v0.4.0, restart live Engram on 3.6.0, live `hermes plugins update recall`.

Route note: all units share `client.py`/`tools.py`, so they are not independent
writers; done inline, test-first, in ~400-line commits.

### Log

- L1 (2026-10-27, owner): "Arranquemos y hagamos todo." — answer to the proposed
  list: compact search, observation history, atomic consolidation, batch judgments +
  atomic session finalization, capture provenance + explicit directory/cwd, protocol
  alignment (canonical topics, compaction recovery outside the system prompt), after
  upgrading Engram to 3.6.
- L2 Evidence: Engram 3.6.0 refuses cloud autosync to the owner's
  `http://192.168.68.198:18080` (bearer token requires HTTPS); same check exists in
  3.4.0, so it was already off. Out of scope; reported to the owner.
- L3 Decision: the schema payload is 4339 bytes before v0.4.0; three new tools
  (`mem_consolidate`, `mem_judge_many`, `mem_session_finalize`) bring the tool count to
  13. Raise the compactness cap in `test_total_schema_payload_stays_compact` to 5600
  bytes, keep every description <= 80 chars, and assert the new names in order after
  `mem_unpin`.
- L4 Route: parent context hit the 150k backstop; T12-T17 delegated to one
  gentle-ai-worker (sequential, shared files). T18 stays with the parent.
- L5 Decisions (T12, worker): S8 applies to every tool through `_engram_error`, not
  only reads: any HTTP answer is reported as `Engram refused the <action>: <reason>`
  with Engram's `code` plus `available_projects` / `candidate_project` / `remedy`
  when sent; the `engram serve` hint stays only when no server answered. "Route
  missing" = 405, or 404 whose body is not a JSON object (Go ServeMux); a JSON 404 is
  a real refusal. History: each version's content is capped at 1000 chars
  (`HISTORY_CONTENT_MAX_CHARS`, 50 versions per page); `history_cursor` is only
  validated and sent with `include_history` (the server ignores it otherwise); an
  Engram that answers without `history` gets a `history_note` naming >= 3.5.1.
  `mem_judge_many` validates every item locally (like `mem_judge`) and refuses the
  whole batch, naming the index, before sending anything. The T12 commit exceeds
  ~400 lines (about half are tests) because S2+S4+S8 share the client error-body
  change.
- L6 Decisions (T13, worker): with 12 tools the payload hit 6048 bytes, so the new
  schemas use bare `{"type": ...}` params where the name says it all, and
  `mem_judge_many` carries the item shape in its description instead of an item
  schema (relations are validated locally, as `mem_judge` does). The L3 cap of 5600
  is kept. `mem_consolidate` adds `last_action: "consolidated"` to `state.json` (an
  additive value; `hermes_recall/state.py` lists the values in its docstring but
  was outside this worker's edit surface). A confirmation is "outcome unknown" on
  a timeout or any 5xx; a preview failure is reported like any other refusal.
- L7 Decisions (T14, worker): `/end` is suppressed for a runtime session after a
  finalize that ended it or may have (matching 200 receipt, timeout, 5xx, a 200
  whose receipt does not match like gentle-engram's check, 409
  `session_already_ended`); a definitive 400/403/404, a refused connection or an
  older server keeps the normal `/end`. Each suppressing outcome forgets the
  registration like an end, and the next validated registration (a continuation)
  clears the mark so that continuation is ended later. A failed pre-registration
  sends no finalize. `state.json` records the summary id as `session_summary`
  (no new `last_action` value). A tool `mem_save` after a finalize still writes to
  the runtime id without pre-registering, as after `on_session_end` since T11;
  background writes pre-register and land in the continuation. Schema payload with
  13 tools: 5564 bytes.
- L8 Decisions (T15, worker): `pinned` is surfaced in a tool hit only when true
  (an unpinned hit carries no key, saving tokens); `state`, `truncated`,
  `review_after` and `relations` from the compact projection are not forwarded.
  The snippet comes from `content` (legacy), else `preview`; the client unwraps a
  compact answer whose `results` is not a list to `[]`.
- L9 Decisions (T16, worker): provenance is `{source, directory}` only (no
  `raw_project`, as S6 specifies). `directory` is omitted only when no working
  directory can be read. The project override maps to `config` as S6 says; note
  that Engram documents `config` as "derived from .engram/config.json", so
  `explicit_override` would be the closer upstream term — left for T18 review.
  The compaction archive and both capture tool saves carry it too; a non-string
  `project_source` is treated as missing.
- L10 Decisions (T17, worker): the static system prompt grows from 1235 to about
  2400 chars, so its budget test moves from 1500 to 2600. The workflow is written
  as bullets, not numbered steps, because the block must carry no digits (prompt
  cache guard). `mem_session_finalize` is suggested only when the user closes or
  hands off the session; ordinary completion still uses `mem_session_summary`.
  README keeps "tested with 2.0.0 and 3.0.0" until T18's live 3.6.0 smoke.

- L11 Decision (T18, parent): `memory.recall.project` provenance is `explicit_override`,
  not `config`: upstream `internal/project/detect.go` defines `config` as derived from
  `.engram/config.json` and `explicit_override` as the caller supplying a project name.

- L12 Evidence (T18, parent): isolated Engram 3.6.0 smoke (copy of the live DB, port
  17437) through the real provider: compact mem_search + prefetch, history page,
  consolidate preview / 409 fingerprint_mismatch / confirm (#797, sources hidden from
  search), judge-many outcomes, unknown_project refusal with available_projects,
  finalize (#798, ended_at set), session provenance `git_remote`. Found: a tool
  `mem_save` after finalize wrote into the ended root session; fixed test-first in
  95f224a (now `<id>:resume:2`).
- L13 Delivery: as v0.2/v0.3, the branch fast-forwards to main and is tagged v0.4.0.
  Slices are the per-task commits bbc981f..95f224a (~2400 authored lines, ~55% tests);
  each commit is one reviewable unit.

- L14 Evidence (T18, parent): `tests/test_server.py` spawn tests fail intermittently
  under load (fake engram binary misses the 3s `instance-id` probe): 1 of 4 full runs
  on `main` (v0.3.0) in an isolated worktree, 2 of ~7 on the branch; always green
  alone. Pre-existing timing flake, not a v0.4.0 regression; follow-up.

- L15 Release (T18, owner-approved): main fast-forwarded to 3b82722, tag v0.4.0
  pushed, CI and Release green (4 assets). Live Engram was already 3.6.0 (restarted
  2026-10-28 17:43, 793 observations). `hermes plugins update recall` -> 0.4.0 at
  3b82722; `memory status` shows recall active, `recall status` reachable. The
  running Hermes gateway loads it after the owner's /restart.

## v0.5.0 — Proxy Engram's official MCP tools (full parity)

recall stops hand-writing `mem_*` tools. It runs its own `engram mcp --tools=agent`
(stdio, stdlib client), publishes the upstream tool schemas as they are, and injects
the Engram session and project it owns into every call. recall keeps owning the
phases (recall, capture, session lifecycle, compaction, server auto-start). Evidence:
spike 2026-10-28 (Engram memory #794, topic `hermes-recall/mcp-proxy-architecture`).

### Specs

- S9 Tool source: the model gets exactly the tools `engram mcp --tools=agent` lists
  (23 on Engram 3.6.0), with upstream names, descriptions and input schemas mapped to
  Hermes `{name, description, parameters}`. Default is all of them ("Sí, los 23
  completos por defecto"). `memory.recall.tools` may hold an allowlist of names.
  `get_tool_schemas` stays cheap and side-effect free on repeat calls (cache).
- S10 Proxy calls: `handle_tool_call` forwards `tools/call` to one long-lived MCP child
  owned by the provider, serialized, with a per-call timeout; the result is the
  upstream result (structured content or text) as JSON; `isError` becomes
  `{"error": ...}` with upstream's text. A dead child is restarted at most once per
  provider instance. The child is stopped on `shutdown`; `engram serve` is never
  stopped.
- S11 Injection (spike map): `session_id` = recall's effective Engram session
  (registered first, like v0.4 saves) on every tool whose schema has `session_id`;
  `project` = detected project when the schema has `project` and the model gave none;
  `cwd` = working directory for `mem_current_project` when absent. A model-supplied
  `session_id` is replaced. When the model names a different `project` on a write,
  `session_id` is not injected (no `session_project_mismatch`). `expected_project`
  is never injected. `mem_session_start` / `mem_session_end` are bound to recall's
  session (inspect their schemas; decide and log).
- S12 Write gate: subagent, cron and flush contexts may call only read tools (an
  explicit read allowlist); any other or future tool counts as a write and is refused
  without starting or calling the child.
- S13 Side effects kept from v0.4: `🧠 saved #id`-style status and `state.json` last id
  from results that carry an observation id; a successful or uncertain
  `mem_session_finalize` and an MCP `mem_session_end` suppress the hook's `/end` and
  the next write resumes (`:resume:N`); private tags redacted before forwarding
  free-text write fields.
- S14 Fallback: when the proxy cannot serve (no `engram` binary, `ENGRAM_URL` set or
  `memory.recall.url` not the local default — the MCP opens the local SQLite, so a
  remote server must never be paired with it —, foreign instance, child fails to
  start or list tools), recall serves the v0.4 HTTP tool set unchanged.
  `hermes recall status` / `/recall` say which source is active.
- S15 Docs: README tools section, design doc, install prompt (Engram MCP in Hermes is
  now redundant: recommend `hermes mcp remove engram`), CHANGELOG 0.5.0, version bump.

### Tasks

- [x] T19 S9+S10 stdio MCP client + schema discovery/cache + proxy dispatch, child
      lifecycle. Route: delegated (gentle-ai-worker), test-first with a fake MCP child.
- [x] T20 S11+S12+S13 injection, write gate, side effects. Route: same worker.
- [x] T21 S14 fallback + status line. Route: same worker.
- [x] T22 S15 docs, CHANGELOG, version 0.5.0. Route: same worker.
- [ ] T23 Verify (independent: concurrency + write gate are high risk), real Engram
      3.6.0 isolated smoke of all 23 tools' binding, release v0.5.0, live install,
      remove the Engram MCP from Hermes. Route: parent.

### Log

- L16 (2026-10-28, owner): "Una consulta pero dime una cosa porque es que no tenemos
  ninguno de los comandos de MCP de Engram? en hermes-recall? se supone que debería
  conectarse a engram y hacer todo esto. no veo porque no tener ese poder en hermes.
  Es un agente igual y debería hacer esos llamados de "mem_*" y hermes-recall solo
  debe asegurarse de que en que fase debe hacerse."
- L17 (owner): "gentle-engram cambia cada semana y trae cosas nuevas. No quiero dejar de
  usarlo. hay otros mantainers que manejan eso. [...] yo quiero tener paridad completa
  de uso. no quiero construir Gentle Engram dentro de hermes-recall."
- L18 Evidence: Hermes' own Engram MCP cannot bind sessions (one process per Hermes,
  fixed cwd, no TERMINAL_CWD, no per-call arg rewrite). Spike: injected
  `session_id`+`project` lands in the HTTP-registered session; plain MCP lands in
  `manual-save-<cwd basename>` of the wrong project. Agent profile: 23 tools, 30.4 KB.
- L19 (owner, choice): "Sí, los 23 completos por defecto" — proxy, all tools by default.
- L20 Decision (T19, worker): Hermes calls `get_tool_schemas` in `add_provider`, before
  `initialize` (agent_init.py `_init_memory`, memory_manager.py `add_provider`), and
  builds its tool routing table from that first answer. So discovery is synchronous on
  the first call: a short-lived child (spawn, `tools/list`, stop), cached in-process per
  binary and on disk in `<hermes_home>/recall/mcp-tools.json` (home from the host's
  `get_hermes_home` before `initialize`). The cache key is the binary's resolved path,
  size and mtime rather than the version string, which would cost a process on every
  start; the version (from `serverInfo`) is stored beside it. Failures are cached
  in-process only. The tool source is fixed per provider instance.
- L21 Decision (T19, worker): the call child starts on the first proxied call, in the
  working directory (`TERMINAL_CWD`, else cwd) so Engram's own cwd fallback matches the
  session, its own session on POSIX (a Ctrl-C on the turn must not kill it), the
  inherited environment unchanged, stderr to `<hermes_home>/recall/engram-mcp.log`.
  Bounds: start 5 s, call 20 s, stop 1 s; a caller waits for the lock at most
  start + call + 2 x stop. A watchdog kills a silent child (also unblocking a write on a
  full pipe). A child found dead before a call is restarted (the one restart); a call
  that died after it was sent is never retried and says it may have been applied. Once
  the restart is spent, proxied calls return an error (no per-call HTTP fallback).
- L22 Decision (T19, worker): result mapping. `structuredContent` wins; else the text
  items joined, as a JSON object when they parse as one, else `{"result": text}`. An
  `isError` JSON body becomes `{"error": message, ...its other fields}` (`error_code`,
  `hint`, `available_projects`); a JSON-RPC error becomes
  `{"error": "Engram MCP refused <tool>: <message>"}`.
- L23 Decision (T19, worker): `memory.recall.tools` takes a YAML list or a
  comma-separated string, ignores names Engram does not list, keeps upstream order, and
  also filters the HTTP fallback tools.
- L24 Delivery (T19): the work unit is two commits over the ~400-line guide:
  45fa1cd (discovery, 1037 lines: 365 are the fake MCP child, 118 tests) and 9737aa9
  (dispatch and lifecycle, 781 lines: ~430 tests).
- L25 Decision (T20, worker): `mem_session_start` is answered by recall and never
  forwarded: the real 3.6.0 handler registers the given id under the project of the
  child's own cwd (`project` "intentionally not read"), while every write is bound to
  recall's session anyway. It registers recall's session (the same `POST /sessions`
  writes use) and returns the effective id. `mem_session_end` is forwarded with `id`
  replaced by the effective session; success or an uncertain outcome suppresses the
  hook's `/end` like a finalize. Both count as writes (S12).
- L26 Decision (T20, worker): `session_id` is injected only after a confirmed
  registration; when none can be confirmed the call goes out without it (the real child
  refuses an unknown id with `unknown_session`, and without one Engram writes to the
  project's own manual-save session). An `unknown_session` refusal re-registers and
  retries once. "Another project" compares Engram's canonical names (NFC, lower-case,
  trimmed, collapsed `--`/`__`). `project` is not injected when `all_projects` is
  truthy (Engram refuses the combination on `mem_search_prompts`); `cwd` only when
  neither `cwd` nor `directory` was given.
- L27 Decision (T20, worker): S12 read allowlist = `mem_search`, `mem_search_prompts`,
  `mem_context`, `mem_get_observation`, `mem_current_project`, `mem_list_projects`,
  `mem_suggest_topic_key`, `mem_doctor` (the tools Engram 3.6.0 marks `readOnlyHint`;
  the hint itself is not trusted). Subagent/cron/flush providers may start their own
  child for these reads (it opens the same store, is stopped on their shutdown, and
  Hermes builds subagents with `skip_memory`, so in practice cron and flush).
- L28 Decision (T20, worker): S13 effects. Private tags are redacted recursively in every
  string of a write's arguments, and only strings carrying a tag are touched. The status
  line goes under `status`, or `recall_status` when Engram's payload already has
  `status` (`mem_consolidate`): the upstream result is never altered. Ids come from the
  result (`id`, finalize `summary_id`, consolidation survivor) or, for `mem_update` and
  `mem_get_observation`, from the arguments. A finalize for another project (no session
  injected) leaves recall's session open. Proxied tool names join the passive-capture
  skip list. T20 is one commit of ~900 lines (~560 tests).
- L29 Decision (T21, worker): "local default" is the address auto-start would spawn for
  (`server.is_spawnable_url`: `http://127.0.0.1|localhost:<ENGRAM_PORT or 7437>`, no
  path); `ENGRAM_URL` set always means fallback. A foreign instance is only known after
  `initialize` (the `/health` identity check runs on the worker), when the host has
  already routed the published tools, so the source cannot change then: every tool call
  is refused with the foreign reason before the child is touched, which is what the v0.4
  tools do against a foreign server. A stale `server.json` is not read at schema time.
- L30 Decision (T21, worker): the decision is recorded in
  `<hermes_home>/recall/tools.json` (`source`, `reason`, `count`, `version`, `label`)
  when it is made, so `hermes recall status` (another process) can name it; `/recall`
  uses the live catalog. The status block gains a `tools:` line only when a source is
  known or asked for, so the v0.4 layout is otherwise unchanged.
- L31 Decision (T22, worker): `provides_tools` stays a static list (Hermes reads it only
  for the plugin catalog and `plugins doctor`; memory-provider tools are routed from
  `get_tool_schemas`). It names every tool the provider can publish on a default
  install: Engram 3.6.0's 23 agent tools in upstream order, then `mem_timeline`, which
  only the HTTP fallback serves; the packaging test pins that list to the fake MCP
  child's 3.6.0 copy. `memory.recall.tools` is declared `type: list, default: []`.
- L32 Evidence (T22, worker): against the real Engram 3.6.0 binary with an isolated
  `ENGRAM_DATA_DIR` and `HERMES_HOME`, `mcp_proxy.discover` took 0.055 s, listed 23
  tools (version 3.6.0) and mapped them to 26,783 bytes of Hermes schemas (30.4 KB
  raw with annotations); a second call hit the in-process cache. The README and
  CHANGELOG quote ~27 KB against ~5.5 KB for the v0.4 tools.
- L33 Process note (T19, worker): to split T19 into two commits, a copy of
  `hermes_recall/mcp_proxy.py` was kept at `/tmp/mcp_proxy_full.py` while the reduced
  first half was committed; it holds only that repository source and was left in
  place (deleting is outside the worker's permissions).

- L34 Evidence (T23, parent): independent verify PASS S9-S15, no blockers/majors.
  Real Engram 3.6.0 smoke through the provider (isolated data dir, cwd /tmp,
  TERMINAL_CWD=repo): 23 tools / 26.8 KB; a model-picked `session_id` was replaced by
  recall's; `<private>` redacted before forwarding; finalize then save landed in
  `<id>:resume:2`; MCP `mem_session_end` closed it; subagent write refused, read
  served; no child left after shutdown. Fixed test-first the verifier's minor: a call
  racing `shutdown()` could spawn a child after `_stop_mcp` (`_mcp_proxy` now checks
  `_shutting_down` under `_mcp_lock`). Spike memory #794 was overwritten by a worker
  upsert; restored as #796 (`hermes-recall/mcp-proxy-spike`).

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
- v0.3.0 released: `48c56e8` + `2fe5a10`, tag v0.3.0, CI and release green
  (4 assets). Live install: `hermes plugins update recall` on the default profile
  -> 0.3.0 at `2fe5a10`; `memory status` shows recall active, `recall status`
  shows 7437 reachable and adopted (Engram 2.0.0). `plugins doctor` warns that the
  manifest's mem_* tools are not added by register(); the manifest only changed
  its version since v0.2.0, and the tools come from the memory provider. Waiting
  for the owner's /restart; then Engram can move to v3 (brew stable is 2.2.0).
- T12 history, mem_judge_many, refusal-aware errors: `bbc981f` (Pi gentle-ai-worker,
  test-first).
- T13 mem_consolidate: `8d78676` (Pi gentle-ai-worker, test-first).
- T14 mem_session_finalize + /end suppression: `2f7e74a` (Pi gentle-ai-worker,
  test-first).
- T15 compact search: `f8da93e` (Pi gentle-ai-worker, test-first).
- T16 capture provenance: `1b6c910` (Pi gentle-ai-worker, test-first, own commit).
- T17 protocol text, docs, CHANGELOG, v0.4.0 bump: `a8f3d14` (Pi gentle-ai-worker,
  test-first). Branch `feat/engram-3.6-additions`: 828 tests, ruff clean; not
  pushed or tagged. T18 (live 3.6.0 smoke, review, release) is next.

- T19 stdio MCP client, discovery and schema cache, proxy dispatch, child lifecycle:
  `45fa1cd` + `9737aa9` (Pi gentle-ai-worker, test-first with a fake MCP child).
- T20 injection, write gate, side effects: `fcf7aba` (Pi gentle-ai-worker, test-first).
- T21 S14 fallback and tool-source status: `ba0d89c` (Pi gentle-ai-worker, test-first).
- T22 docs (README, design, install prompt), CHANGELOG 0.5.0, version bump,
  `provides_tools`, `memory.recall.tools` in the manifest: `c4c29bb` (Pi
  gentle-ai-worker, test-first for the packaging invariants). Branch
  `feat/mcp-proxy`: 918 tests, ruff clean; not pushed or tagged. T23 (independent
  verify, real 3.6.0 smoke of all 23 bindings, release, live install, remove the
  Engram MCP from Hermes) is next.

## Next step

Owner sends /restart and checks the 🧠 line and /odd_status. v0.2.x follow-ups:
review advisories (server.py docstrings, "unavailable" overloaded, tools.py:619),
pre-registration outage cost, delegation non-str guard, pip-install CLI gap;
optionally remove the Engram MCP once recall is field-proven.
