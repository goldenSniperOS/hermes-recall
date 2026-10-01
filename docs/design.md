# hermes-recall design

`hermes-recall` is an exclusive Hermes Agent memory provider (`memory.provider: recall`)
that makes a local Engram HTTP server (`engram serve`, default
`http://127.0.0.1:7437`) the agent's memory. Runtime is Python stdlib only.

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
| `__init__.py` (root) | Loader entry point; re-exports `register(ctx)` |
| `cli.py` (root) | `hermes recall status` / `hermes recall show <id>` |

### Hooks and tools → Engram endpoints

| Hermes surface | Engram endpoint | Notes |
| --- | --- | --- |
| `initialize` | `GET /project/current?cwd=`, `POST /sessions` | Background; primary context only; retried by later writes if it fails |
| `prefetch` | `GET /search`, `GET /context` (once per session) | Trivial prompts skipped; 4 s budget |
| `queue_prefetch` | `GET /project/current` (until detected) | Warms project detection only; no search, never consumes the `/context` flag |
| `recall_status` | — | `🧠 Recall · <project> — recalled N memories`, rendered by Hermes |
| `on_turn_start` | `POST /prompts` | `<private>` redacted, 2000-char cap |
| `sync_turn` | `POST /observations/passive` | Tool results + assistant text, ≤ 12 per turn; skips `mem_*`, `mcp__engram__*`, `mcp_engram_*` results |
| `on_delegation` | `POST /observations/passive` (`source=subagent`) | Task / result redacted, then capped at 1000 / 4000 chars |
| `on_pre_compress` | `POST /observations` (`session_summary`), `GET /context/compaction` | Synchronous, 7 s budget, idempotent by hash |
| `on_session_switch` / `on_session_end` | `POST /sessions`, `POST /sessions/{id}/end` | `/end` de-duplicated |
| `mem_search` | `GET /search` | Default project = detected project |
| `mem_save` | `POST /observations` | Registers the session and retries once on `session not found` |
| `mem_update` | `PATCH /observations/{id}` | Only given fields are sent |
| `mem_get_observation` | `GET /observations/{id}` | |
| `mem_context` | `GET /context` | |
| `mem_session_summary` | `POST /observations` (`type=session_summary`) | |
| `mem_timeline` | `GET /timeline` | |
| `mem_judge` | `POST /conflicts/judge` | `relation` ∈ related, compatible, scoped, conflicts_with, supersedes, not_conflict |
| `mem_pin` / `mem_unpin` | `PUT` / `DELETE /observations/{id}/pin` | |
| `/recall`, `hermes recall status` | `GET /health` | Plus `state.json` |
| `/recall <id>`, `hermes recall show <id>` | `GET /observations/{id}` | |

### Tool results

Every tool returns a JSON string and never raises. Failures are `{"error": "..."}`
(unknown tool, bad arguments, writes refused, Engram unreachable or rejecting).

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

`mem_save` passes every key of Engram's response through (Engram's own `status`
is renamed `engram_status`). When the response carries judgment data
(`judgment_required`, `judgment_id`, `candidates`, `conflicts`), the result also
gets a `next_step` hint telling the model to call `mem_judge`. Engram 2.0's HTTP
`POST /observations` (`handleAddObservation`, `server.go`) answers only
`{"id", "status"}` today, so this is forward compatibility, not a current path.

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
  | `last_action` | str | `saved`, `updated`, `observation`, `pinned`, `unpinned`, `session_summary` |
  | `project` | str | Engram project of that action (may be `""`) |
  | `updated_at` | str | ISO 8601 UTC, e.g. `2026-01-02T03:04:05Z` |

- **Atomicity:** written to a temp file in the same directory, `fsync`ed, then
  `os.replace`d over the target. Readers never see a partial file.
- **When:** after every successful `mem_save`, `mem_session_summary`,
  `mem_update`, `mem_get_observation`, `mem_pin`, `mem_unpin` in the primary agent
  context. Write failures are swallowed; the tool result is unaffected.
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
  `mem_judge`, `mem_pin`, `mem_unpin`) return an error. Read tools work and do not
  write `state.json`.

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
| `mem_context` text | 6000 chars | `tools.CONTEXT_MAX_CHARS` |
| `mem_timeline` before/after | 5 default, 20 max | `tools.TIMELINE_WINDOW_MAX` |
| `/recall <id>` content | 800 chars | `format.DETAIL_CONTENT_MAX_CHARS` |
| Worker queue | 256 jobs | `provider._QUEUE_MAXSIZE` |
| Shutdown drain | 2 s | `provider.SHUTDOWN_DRAIN_TIMEOUT_S` |
