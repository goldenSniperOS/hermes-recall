# hermes-recall design

`hermes-recall` is an exclusive Hermes Agent memory provider (`memory.provider: recall`)
that makes a local Engram HTTP server (`engram serve`, default
`http://127.0.0.1:7437`, or the `ENGRAM_PORT` port) the agent's memory. Runtime is Python stdlib only.

## Architecture

```
Hermes MemoryManager ──hooks──▶ RecallProvider (provider.py)
                                   │  recall / capture / lifecycle
                                   │  tools ──▶ tools.py ──▶ state.py (state.json)
                                   ▼
                              EngramClient (client.py) ──HTTP──▶ Engram
```

| Module | Role |
| --- | --- |
| `hermes_recall/provider.py` | `RecallProvider`: every `MemoryProvider` hook, the background worker, `/recall` handler |
| `hermes_recall/client.py` | Fail-soft HTTP client: short timeouts, `None`/`[]`/`""` on failure, per-thread `last_error_text()` |
| `hermes_recall/tools.py` | The 10 `mem_*` schemas, their handlers, and the shared status/detail reports |
| `hermes_recall/state.py` | The stable `state.json` contract (`read_state`, `write_state`) |
| `hermes_recall/format.py` | Pure rendering: recall block, status block, observation detail, `🧠` status strings |
| `hermes_recall/config.py` | Settings (`ENGRAM_URL` > `memory.recall.*` > defaults), Hermes home resolution |
| `hermes_recall/server.py` | Engram server lifecycle: adopt a verified server or start a detached `engram serve`; `server.json` |
| `__init__.py` (root) | Loader entry point; re-exports `register(ctx)` |
| `cli.py` (root) | `hermes recall status` / `hermes recall show <id>` |

### Hooks and tools → Engram endpoints

| Hermes surface | Engram endpoint | Notes |
| --- | --- | --- |
| `initialize` | `GET /health` (+ `engram instance-id`, maybe `engram serve`), `GET /project/current?cwd=`, `POST /sessions` | Background; primary context only; server ensured before registration; registration retried by later writes if it fails |
| `prefetch` | `GET /search?response_format=compact`, `GET /context` (once per session) | Trivial prompts skipped; 4 s budget; hits render from `content`, else the compact `preview` |
| `queue_prefetch` | `GET /project/current` (until detected) | Warms project detection only; no search, never consumes the `/context` flag |
| `recall_status` | — | `🧠 Recall · <project> — recalled N memories`, rendered by Hermes |
| `on_turn_start` | `POST /prompts` | `<private>` redacted, 2000-char cap |
| `sync_turn` | `POST /observations/passive` | Tool results + assistant text, ≤ 12 per turn; skips `mem_*`, `mcp__engram__*`, `mcp_engram_*` results |
| `on_delegation` | `POST /observations/passive` (`source=subagent`) | Task / result redacted, then capped at 1000 / 4000 chars |
| `on_pre_compress` | `POST /observations` (`session_summary`), `GET /context/compaction` | Synchronous, 7 s budget, idempotent by hash |
| `on_session_switch` / `on_session_end` | `POST /sessions` (`resume: true`), `POST /sessions/{id}/end` | `/end` de-duplicated; ends the effective session id; skipped after `mem_session_finalize` |
| `mem_search` | `GET /search?response_format=compact` | Default project = detected project; accepts the compact `{result, results}` answer and the legacy bare array; `pinned: true` surfaced; snippet from `content` or `preview` |
| `mem_save` | `POST /observations` | Registers the session and retries once on `session not found` |
| `mem_update` | `PATCH /observations/{id}?expected_project=` | `expected_project` required, never defaulted; only given fields are sent |
| `mem_get_observation` | `GET /observations/{id}` (`?include_history=true&history_cursor=`) | History keys lifted out of `observation`; each version's content capped; an Engram without history gets a `history_note` |
| `mem_context` | `GET /context` | |
| `mem_session_summary` | `POST /observations` (`type=session_summary`) | |
| `mem_timeline` | `GET /timeline` | |
| `mem_judge` | `POST /conflicts/judge` | `relation` ∈ related, compatible, scoped, conflicts_with, supersedes, not_conflict |
| `mem_pin` / `mem_unpin` | `PUT` / `DELETE /observations/{id}/pin` | |
| `mem_consolidate` | `POST /observations/consolidate` | Preview without `confirm`; apply with `confirm` + the approved plan's `fingerprint`; `expected_project` required; title/content redacted; never retried, never auto-confirmed |
| `mem_session_finalize` | `POST /sessions/{effective id}/finalize` | Body exactly `{project, summary}`; registers first; sent once; see "Session finalize" |
| `mem_judge_many` | `POST /conflicts/judge-many` | 1..32 judgments, each validated like `mem_judge` before anything is sent; effective session id; never retried |
| `/recall`, `hermes recall status` | `GET /health` | Plus `state.json` |
| `/recall <id>`, `hermes recall show <id>` | `GET /observations/{id}` | |

### Tool results

Every tool returns a JSON string and never raises. Failures are `{"error": "..."}`
(unknown tool, bad arguments, writes refused, Engram unreachable or rejecting).

An Engram that answered with an HTTP error is not an outage: the error reads
`Engram refused the <action>: <reason>` and carries Engram's `code` and, when it
sent them, `available_projects`, `candidate_project` or `remedy` (for example 404
`unknown_project`, 409 `ambiguous_project`). Only when no server answered does the
error end with "Is `engram serve` running?". The Engram 3.6 workflow tools
(`mem_judge_many`, and later ones) have no `/health` capability flag; an older
server answers their route with a plain-text 404 or a 405, which becomes
`<tool> needs Engram >= 3.5.1`.

Results that carry an observation id include a `status` field:

| Tool | `status` | `state.json` `last_action` |
| --- | --- | --- |
| `mem_save` | `🧠 saved #761` | `saved` |
| `mem_session_summary` | `🧠 saved #761` | `session_summary` |
| `mem_update` | `🧠 updated #761` | `updated` |
| `mem_get_observation` | `🧠 observation #761` | `observation` |
| `mem_pin` | `🧠 pinned #761` | `pinned` |
| `mem_unpin` | `🧠 unpinned #761` | `unpinned` |
| `mem_judge` | `🧠 judged <judgment_id>: <relation>` | — |
| `mem_judge_many` | `🧠 judged 3 of 4 (1 failed)` | — |
| `mem_session_finalize` | `🧠 session finalized, summary #950` | `session_summary` |
| `mem_consolidate` (preview) | `🧠 consolidation preview: 2 observations into 1` | — |
| `mem_consolidate` (confirm) | `🧠 consolidated #30 (soft-deleted #11, #12)` | `consolidated` |

`mem_save` passes every key of Engram's response through (Engram's own `status`
is renamed `engram_status`). When the response carries judgment data
(`judgment_required`, `judgment_id`, `candidates`, `conflicts`), the result also
gets a `next_step` hint telling the model to call `mem_judge`. Engram 2.0's HTTP
`POST /observations` (`handleAddObservation`, `server.go`) answers only
`{"id", "status"}` today, so this is forward compatibility, not a current path.

`mem_consolidate` follows Engram's two-step contract. Without `confirm` the server
writes nothing and the result carries the `plan`, its `fingerprint` and a
`next_step` telling the model to show the plan to the user and to confirm only
after they approve it. A 409 `fingerprint_mismatch` (the sources changed since
the preview) is an error carrying the fresh `plan`; it is never confirmed
automatically. A confirmation that timed out or got a 5xx may have been applied,
so the error says the outcome is unknown and asks the model to check the source
ids instead of retrying.

## Engram server lifecycle

Ported from gentle-engram's `initializeEngramServer` (`index.ts`), so Hermes and Pi
share one server under the same rules. `server.ensure_server()` runs once on the
background worker from the primary context's `initialize`, queued **before** session
registration on the same serialized worker. `initialize` returns at once, and
`prefetch` keeps its own bounded path, so the first turn never waits for a start.

### Decision table

Checked in this order; the first matching row decides.

| Condition | Status | Action |
| --- | --- | --- |
| `ENGRAM_URL` set in the environment | `external` | none: externally managed |
| `memory.recall.autostart: false` | `disabled` | none |
| URL is not `http://127.0.0.1:<port>` or `http://localhost:<port>` with `<port>` = `ENGRAM_PORT` or 7437, with no path (other than `/`), query or fragment | `external` | none: a spawned `engram serve` could not answer there |
| `/health` ok, `instance_id` = `ENGRAM_BIN instance-id` | `adopted` | none |
| `/health` ok, different `instance_id` | `foreign` | **no memory traffic** for this provider instance; never managed, never killed |
| `/health` ok, no `instance_id` | `unavailable` | used, never managed: legacy (version < 2.0.0-rc.11) or identity missing (see the decision below) |
| `/health` ok, `instance-id` cannot be resolved | `unavailable` | used, never managed: identity cannot be verified |
| no answer, binary missing or `instance-id` fails | `unavailable` | none ("install Engram or set ENGRAM_BIN") |
| no answer, start lock held by another process | `adopted` / `foreign` / `unavailable` | poll `/health` until the deadline, never spawn; classify whatever answers |
| no answer, lock taken, provider shutting down | `unavailable` | none: no spawn after shutdown |
| no answer, lock taken | `started` / `adopted` / `foreign` / `unavailable` | re-probe, spawn `ENGRAM_BIN serve`, poll `/health` until the deadline; `adopted` when a matching server answers but our child already exited (the port was taken) |

### Decision: only a definite mismatch is gated

When `/health` reports an `instance_id` that differs from `ENGRAM_BIN instance-id`,
the server is provably a different Engram (another store). For the rest of that
provider instance recall sends it **no memory traffic**: `prefetch` and
`queue_prefetch` return nothing, project detection, session registration, prompt,
passive and delegation capture, session end and the compaction archive are skipped,
the `mem_*` tools and `/recall <id>` return an error naming the reason, and the
status line reads `foreign instance — not used: <reason>`. Only the status probe
(`GET /health`) still reaches it. Everything before classification is unchanged.

This **diverges from gentle-engram** on purpose in the other direction: gentle-engram
fails closed for a healthy server with no identity (legacy or identity missing) and
for a binary that cannot resolve `instance-id`. recall still **uses** those servers
and never manages them (`unavailable`, shown as `reachable, not verified —
<reason>`, no restart). Reason: a gateway whose `PATH` lacks the `engram` binary, or
a user still running an older Engram, must not silently lose memory. A mismatch is
evidence of the wrong store; a missing identity is only missing evidence.

Default URL: with neither `ENGRAM_URL` nor `memory.recall.url` set, the URL is
`http://127.0.0.1:<ENGRAM_PORT or 7437>`, exactly gentle-engram's
`CONFIGURED_ENGRAM_URL || http://127.0.0.1:${ENGRAM_PORT}` (`index.ts:24-26`), so the
default always matches what a spawned `engram serve` binds.

Why the port rule: Engram binds `fmt.Sprintf("127.0.0.1:%d", s.port)` (`server.go`
`Start`), and the binary documents `ENGRAM_PORT  Override HTTP server port (default:
7437)`. The child inherits our environment, so it binds exactly `127.0.0.1:<port>`.
`[::1]` is not accepted because the server listens on IPv4 loopback only.

### Bounds

| Step | Bound |
| --- | --- |
| `ENGRAM_BIN instance-id` | 3 s (stdout only; the real binary prints an update notice on stderr) |
| each `/health` probe | 0.5 s |
| readiness polling after a spawn, or while another process holds the lock | 5 s total, every 0.1 s |

### Spawn

`ENGRAM_BIN` (else `engram` on `PATH`) is launched as exactly `[bin, "serve"]` with
`start_new_session=True` (Windows: detached process group), `stdin` from `/dev/null`,
stdout and stderr appended to `<hermes_home>/recall/engram-serve.log` (or discarded
when it cannot be opened) and `close_fds=True`.

- **Environment: inherited** from Hermes. `ENGRAM_PORT` and Engram's own data-dir
  settings must reach the child, or it would bind another port or open another
  store than the one recall (and Pi) talk to. The child gets a copy plus
  `ENGRAM_CLOUD_AUTOSYNC=1`, as gentle-engram 0.2.0 spawns it (`2f1e545`); the
  server skips autosync itself when no cloud server or token is configured. The
  Hermes process environment is not modified, and adopted or external servers are
  untouched.
- **cwd: `<hermes_home>/recall`**, not the Hermes process cwd, so a long-lived
  server never holds a user repository as its working directory.
- **Log cap:** before each spawn, a log over 1 MB is rotated to
  `engram-serve.log.1` (replacing the previous one), then the new run appends.
- **Reaping:** a daemon thread calls `wait()` on the child, so when the server
  later exits on its own it never stays a zombie in a long-lived gateway. This
  only collects the exit status; it never signals the process.

### Cross-process lock

An exclusive non-blocking `fcntl.flock` on `<hermes_home>/recall/serve.lock` wraps
the re-probe, the spawn and the readiness wait, so a gateway and a CLI session
cannot both spawn. A process that cannot take the lock only polls `/health` within
the deadline. Without `fcntl` (Windows), or when the lock file cannot be opened,
the start goes ahead without it.

### One bounded restart

The client fires a refusal listener on a TCP connection refusal only (not on a
timeout or an HTTP error, which mean a server may be alive). When the status is
`adopted` or `started`, the provider schedules one `ensure_server` on the worker.
At most one per provider instance; never for `foreign`, `external` or `disabled`.

### Never kill

Nothing in recall stops a server: not a foreign one, not one that never became
ready, not the one recall started, not at `shutdown`. The server is shared with Pi
and every other agent on the machine. gentle-engram terminates a child it gave up
on during startup; recall deliberately does not (a slow start may still finish and
serve the next session).

### Status

The outcome is kept on the provider and written to `<hermes_home>/recall/server.json`
(`status`, `reason`, `url`, `updated_at`), so `hermes recall status` in another process
can show it. The `server:` line combines the live `/health` probe with the outcome:
`reachable, started by recall`, `reachable, adopted`, `reachable, foreign instance —
not used: <reason>`, `reachable, not verified — <reason>`, `reachable, externally
managed`, `reachable, autostart off`, or `not running — <reason>`. Without a recorded outcome the line is just
`reachable` / `unreachable`. `server.json` is internal, not a stable contract.

## `state.json` contract (stable)

Other plugins depend on this file (hermes-odd reads `last_memory_id`). It is a
**stable** contract: changes require a major version.

- **Path:** `<hermes_home>/recall/state.json`. `hermes_home` is the `hermes_home`
  kwarg Hermes passes to `initialize` (the active profile's home); without it,
  `$HERMES_HOME`, then `~/.hermes`.
- **Content:** one JSON object with exactly these keys:

  | Key | Type | Meaning |
  | --- | --- | --- |
  | `last_memory_id` | int | Observation id of the last successful id-yielding tool call |
  | `last_action` | str | `saved`, `updated`, `observation`, `pinned`, `unpinned`, `session_summary`, `consolidated` (added in 0.4.0) |
  | `project` | str | Engram project of that action (may be `""`) |
  | `updated_at` | str | ISO 8601 UTC, e.g. `2026-01-02T03:04:05Z` |

- **Atomicity:** written to a temp file in the same directory, `fsync`ed, then
  `os.replace`d over the target. Readers never see a partial file.
- **When:** after every successful `mem_save`, `mem_session_summary`,
  `mem_update`, `mem_get_observation`, `mem_pin`, `mem_unpin`, confirmed
  `mem_consolidate` (the survivor id) and `mem_session_finalize` (the summary id) in the primary agent context. Write failures
  are swallowed; the tool result is unaffected.
- **API:**

  ```python
  from hermes_recall.state import read_state, write_state

  read_state(hermes_home) -> dict   # {} when missing or corrupt; never raises
  write_state(hermes_home, *, last_memory_id: int, last_action: str, project: str) -> None
  ```

  `read_state` validates types: a file whose values have the wrong types counts
  as corrupt and returns `{}`. `state.py` imports only the stdlib, so another
  plugin can import it without loading the provider.

## Findings

### Visible "saved #id" line

There is no supported way for a memory provider to emit a user-visible line from a
tool call:

- `MemoryProvider.handle_tool_call(tool_name, args, **kwargs) -> str`
  (`agent/memory_provider.py`) only returns the JSON tool result. The ABC defines
  no status callback.
- `MemoryManager.handle_tool_call` (`agent/memory_manager.py`) routes by name and
  returns the provider's string unchanged; it forwards `**kwargs` but adds no
  emitter to them.
- The only deterministic, model-independent indicator is the recall line:
  `MemoryProvider.recall_status()` → `MemoryManager.describe_recall()` →
  `agent._emit_status(...)`, called right after `prefetch_all` on the turn thread
  (`agent/turn_context_prefetch_excerpt.py`). It describes the last prefetch only.

So the `🧠 saved #761` text lives in the tool result's `status` field, where the
model sees it and can repeat it, and the id persists in `state.json` for
`/recall`, `hermes recall status` and other plugins. Core is not monkeypatched.

### `/recall` slash command

For an exclusive provider, `register(ctx)` receives the loader's
`_ProviderCollector` (`plugins/memory/__init__.py`). Its `__getattr__` forwards
every `register_*` call — including `register_command` — to a real
`PluginContext(PluginManifest(name="recall"), get_plugin_manager())`, swallowing
and logging any failure. `PluginContext.register_command(name, handler,
description="", args_hint="", argument_mode=None)` takes a handler
`fn(raw_args: str) -> str | None`.

`hermes_recall.register` therefore registers `/recall` with `args_hint="[id]"`,
guarded by `try/except` so it can never cost the provider registration. It skips
registration when the collector was built with `register_skills=False` (the
loader's inspection mode used by `discover_memory_providers`, `hermes memory
status` and the setup picker), so an installed but inactive provider does not
leak `/recall`. That check reads the collector's `_register_skills` attribute and
defaults to registering when it is absent.

The command is registered at load time, but the agent may initialize a different
instance. The handler reports on the most recently initialized instance (held
weakly) and falls back to the one it was registered with.

### `hermes recall` CLI

`discover_plugin_cli_commands` (`plugins/memory/__init__.py`) only runs for the
active `memory.provider`. It loads `<provider dir>/cli.py`, calls
`register_cli(subparser)` and takes `recall_command` as the handler (it looks up
`<provider name>_command`). For a directory install the provider dir is the
repository root, so `cli.py` lives there. `register_cli` also sets
`func=recall_command` on the subparser, matching the documented example.

The loader also executes every top-level `*.py` in the provider dir as a
submodule when it loads the provider, so `cli.py` keeps its module scope
import-free and imports the package lazily.

**Known gap:** for a pip install, `find_provider_dir` resolves to the package
directory of the entry point (`hermes_recall/`), so the CLI would need a
`hermes_recall/cli.py`. The pip route currently gets the provider and tools but
not `hermes recall`.

## Fail-soft and threading

- No hook or tool raises. Hooks degrade to "no memory this turn"; tools return a
  JSON error.
- `is_available()` makes no network call.
- One daemon worker thread runs every background job (session registration,
  prompt and passive capture, queued project detection, `/end`) in order, behind a
  bounded queue (256). A full queue drops the job. `shutdown()` joins for at most 2 s.
- Tools and `on_pre_compress` run on the caller's thread. `mem_save` may register
  the session synchronously when Engram has not seen it yet.
- `queue_prefetch` does not search. Hermes calls it with the turn that just
  finished, while the next `prefetch` asks about the next message, so a recall
  computed in the background would almost never be served. It only warms project
  detection; the first non-trivial `prefetch` still fetches `/context`.
- `EngramClient.last_error_text()` is per thread, so the worker and the tool
  thread cannot overwrite each other's failure reason.
- Non-primary `agent_context` (`subagent`, `cron`, `flush`): no capture and no
  session writes; write tools (`mem_save`, `mem_update`, `mem_session_summary`,
  `mem_judge`, `mem_judge_many`, `mem_consolidate`, `mem_session_finalize`,
  `mem_pin`, `mem_unpin`) return an error. Read tools work and do not write
  `state.json`.

## Session registration

Engram rejects a write that names a project when it has no row for the session
(`validateSessionProject`, `server.go`: 404 `session not found`). The `POST
/sessions` from `initialize` or a session switch fails whenever Engram is down or
the project cannot be resolved yet, so registration is not a one-shot:

- The provider remembers the session id Engram last confirmed. A switch clears it.
- Before each background write (prompt, passive capture, delegation) and the
  compaction archive, an unconfirmed session is registered first.
- A write that still answers `session not found` registers and retries once,
  through the same helper `mem_save` uses (`tools.with_session_retry`).
- Each job posts `/sessions` at most once, however many captures it carries. At
  compaction, registration only starts when it and the archive write both fit in
  the 7 s budget.

### Effective session id

Registration sends `"resume": true`, as gentle-engram 0.2.0 does (Engram v3,
`6af3a4b`). When the session already ended (Hermes `/resume` into a session recall
ended earlier, or a write after `on_session_end`), Engram v3 picks the lowest live
`<id>:resume:<N>` continuation, or the next ordinal, and acknowledges that id
(`resumed_from` names the root). The choice is deterministic, so registering again
returns the same live continuation. Engram 2.0.0 ignores `resume` and echoes the
id; v2.2 answers 409 `session_already_ended`, which stays a failed registration.

- The ack is validated like upstream: `status` must be `created` and `id` either
  the Hermes session id or `<id>:resume:<N>` with an ASCII-numeric `N`. Anything
  else is a failed registration and its id is never written to, including an ack
  with no `id` or a non-string one (upstream `registerEffectiveSession` rejects
  those too).
- A validated id is the session's **effective** id. The provider maps Hermes id →
  effective id (bounded, 64 entries). Every session-attributed request uses it:
  prompts, passive and delegation capture, `mem_save`, `mem_session_summary`,
  `mem_judge`, the compaction archive, `GET /context/compaction` and `/end`.
  Without a mapping the Hermes id is used, as before.
- Jobs snapshot the Hermes id when queued and resolve the effective id when they
  run, right after their own registration (or the `session not found` retry). The
  map is keyed by Hermes id and survives switches, so a job queued for one session
  can never write into another session's continuation.
- The `/end` job ends the effective id, then drops the mapping and the confirmed
  flag, so the next write for that Hermes id registers again and lands in a
  continuation. Switching into a session removes it from the recently-ended set,
  so a later `/reset` ends its continuation too.
- That removal happens on every switch. On Engram 2.0.0, which has no
  continuations, end(A) → switch(B) → resume(A) → `/reset` therefore posts a
  second `/end` for A. This is intentional and harmless: 2.0.0's `EndSession` just
  re-runs its `UPDATE`, answering 200 and moving `ended_at` to the second end (the
  summary stays empty, as both ends send one).

### Session finalize

`mem_session_finalize` (Engram >= 3.5.1) saves the summary and ends the session in
one transaction: `POST /sessions/{effective id}/finalize` with exactly
`{project, summary}`. An unconfirmed session is registered first; if that fails,
nothing is posted. The request is sent once:

| Outcome | Tool result | Shutdown / reset `/end` |
| --- | --- | --- |
| 200 `finalized` / `already_finalized` with a matching receipt | `🧠 session finalized, summary #<id>` | Not posted |
| Timeout, 5xx, or a 200 whose receipt does not match | "outcome is unknown: do not retry" | Not posted (no fallback) |
| 409 `session_already_ended` | Error with the code | Not posted |
| 400 / 403 `session_project_mismatch` / 404 `unknown_session`, refused connection, older server | Error with the code (or "needs Engram >= 3.5.1") | Posted as usual |

Every "not posted" outcome also forgets the session's registration, exactly as an
end does, so the next write registers again and, with `resume`, lands in a live
continuation; that registration clears the finalized mark, so the continuation
is ended normally later.

## `mem_update` ownership

Engram v3 (`e8c0ce3`) requires `?expected_project=` on `PATCH /observations/{id}`:
400 when missing or invalid, 409 when it does not match the observation's owner
or the patch would move it. The server treats it as an explicit caller assertion,
so the tool schema makes it required and recall never fills it in from the
detected project. A blank value is refused locally with no request. Any HTTP
error answer (the client's `last_error_status` is set) comes back as `Engram
refused the update of #<id>: ...` with the server's reason, without the "is
`engram serve` running?" hint. A transport failure (refused, unreachable, timed
out) keeps the hint. Engram 2.0.0 through 2.2.1 never read the query string in
`handleUpdateObservation` (it calls `store.UpdateObservation(id, body)`), so
there the parameter has no effect.

## Capture high-water mark

`sync_turn` receives the whole transcript every turn and captures only messages
past a high-water mark. After startup or a session switch (`/resume`, `/new`,
compression) the next transcript may carry history from an earlier session, so
the first `sync_turn` starts at the last user message: only the turn that just
ran is captured. A genuinely new transcript has its only user message first, so
nothing of it is skipped. A rewind keeps the mark.

## Privacy

- `<private>…</private>` blocks are redacted on the full raw text **before** any
  truncation: prompts (2000), delegation task / result (1000 / 4000) and every
  compaction digest row (200 per user turn, 800 for the last reply). A cut can
  therefore never split a block and store its head. Passive tool results are
  redacted and not truncated.
- For a loopback Engram URL (`127.0.0.1`, `::1`, `localhost`) the client's opener
  has an empty `ProxyHandler`, so `HTTP_PROXY` / `HTTPS_PROXY` never see memory
  content. A remote URL keeps the environment's proxies.

## Limits and caps

| Limit | Value | Where |
| --- | --- | --- |
| HTTP read / write timeout | 2 s / 3 s | `client.py` |
| Synchronous prefetch budget | 4 s | `provider.PREFETCH_BUDGET_S` |
| Compaction budget | 7 s | `provider.PRE_COMPRESS_BUDGET_S` |
| Recall search query | 200 chars | `provider.QUERY_MAX_CHARS` |
| Recall hits | 5 (1..20) | `memory.recall.recall_limit` |
| Recall block | 2500 chars (200..20000) | `memory.recall.recall_max_chars` |
| Prompt capture | > 10, ≤ 2000 chars | `provider.PROMPT_*` |
| Passive captures per turn | 12, each > 50 chars | `provider.PASSIVE_*` |
| Compaction digest / recovery text | 4000 / 1500 chars | `provider.PRE_COMPRESS_MAX_CHARS`, `RECOVERY_MAX_CHARS` |
| `mem_search` limit | 10 default, 20 max | `tools.SEARCH_LIMIT_*` |
| `mem_search` query / snippet | 500 / 300 chars | `tools.QUERY_MAX_CHARS`, `SEARCH_SNIPPET_MAX_CHARS` |
| `mem_get_observation` content | 12000 chars | `tools.OBSERVATION_CONTENT_MAX_CHARS` |
| `mem_get_observation` history version | 1000 chars each, 50 per page (Engram) | `tools.HISTORY_CONTENT_MAX_CHARS` |
| `mem_judge_many` batch | 1..32 | `client.JUDGE_MANY_MAX` |
| `mem_context` text | 6000 chars | `tools.CONTEXT_MAX_CHARS` |
| `mem_timeline` before/after | 5 default, 20 max | `tools.TIMELINE_WINDOW_MAX` |
| `/recall <id>` content | 800 chars | `format.DETAIL_CONTENT_MAX_CHARS` |
| Worker queue | 256 jobs | `provider._QUEUE_MAXSIZE` |
| Shutdown drain | 2 s | `provider.SHUTDOWN_DRAIN_TIMEOUT_S` |
