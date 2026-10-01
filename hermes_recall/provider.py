"""``RecallProvider``: the Hermes ``MemoryProvider`` hooks (recall, capture, lifecycle)."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import client as client_module
from ._hermes import MemoryProvider, RecallStatus, is_trivial_prompt
from .client import EngramClient
from .config import RecallConfig, load_config
from .format import render_recall_block
from .redact import strip_private, truncate

logger = logging.getLogger(__name__)

# Engram's FTS query is a search string, not a prompt. A long paste would blow the
# query apart (and the URL) for no ranking benefit, so the turn's text is trimmed.
QUERY_MAX_CHARS = 200

# Total wall-clock ceiling on the synchronous prefetch path.
#
# Hermes joins an external provider's prefetch for 8s, and on timeout it leaves the
# thread running and registers it: every later turn that finds it still alive logs
# "prefetch is still running; skipping this turn" and returns "" immediately. One
# overrun therefore kills recall for the REST of the session, not just for the turn
# that blocked. The client's per-call 2s read timeout bounds *one* call; this bounds
# their SUM, which is the number the host actually cares about.
#
# The arithmetic (asserted in tests/test_provider.py):
#   * Two calls are mandatory — project detection and the search — and each is
#     bounded by client.READ_TIMEOUT_S (2s). They are gated on "has the deadline
#     passed", because skipping the search means no recall at all, so the budget
#     must never be the reason the feature does nothing. Worst case: 2 + 2 = 4s.
#   * The once-per-session /context call is optional, so it is gated harder: it only
#     starts when the remaining budget can absorb a whole read timeout. It can
#     therefore never push the total past the deadline.
# Hence the worst-case synchronous path is PREFETCH_BUDGET_S itself, which is why
# this must stay >= 2 * READ_TIMEOUT_S (or the budget would cut off recall) and
# comfortably below the host's 8s join.
PREFETCH_BUDGET_S = 4.0

# Prompt capture. Upstream gentle-engram posts ``/prompts`` only when the trimmed
# prompt is longer than 10 characters (``finalContent.length > 10``, index.ts) and
# caps it at 2000 (``truncate(finalContent, 2000)``). Both adopted for parity: the
# floor keeps "ok then" out of the prompt log, the cap keeps a pasted stack trace
# from becoming the user's remembered words.
PROMPT_MAX_CHARS = 2000
PROMPT_MIN_CHARS = 10

# Passive capture. Upstream skips a tool result of 50 characters or fewer
# (``content.length <= 50``, index.ts ``tool_execution_end``) and imposes no upper
# bound on passive content — both adopted.
PASSIVE_MIN_CHARS = 50

# ...but upstream captures one tool result per event, whereas we are handed the whole
# turn at once. A 200-tool-call turn would otherwise become 200 serialized HTTP
# writes on a single worker, which is how a background queue turns into a backlog
# that outlives the turn it belongs to. Cap the posts per turn instead.
PASSIVE_MAX_CAPTURES_PER_TURN = 12

# Delegation capture. A subagent transcript is unbounded — it is a whole agent run —
# so both halves are capped before they are composed. The result gets the larger
# share because it is the part that carries findings; the task is a prompt, and a
# prompt that needs more than 1000 characters to identify is already summarised by
# its first 1000.
DELEGATION_TASK_MAX_CHARS = 1000
DELEGATION_RESULT_MAX_CHARS = 4000

# Compaction. The archived digest is a *summary* of the window being discarded, never
# the raw transcript — the transcript is exactly what compaction exists to shed.
PRE_COMPRESS_MAX_CHARS = 4000
# The recovery text is prepended to the compression summary prompt, so it competes for
# the very context budget compaction is trying to reclaim. Keep it short.
RECOVERY_MAX_CHARS = 1500
_DIGEST_USER_TURNS = 10
_DIGEST_USER_CHARS = 200
_DIGEST_ASSISTANT_CHARS = 800

# Total wall-clock ceiling on on_pre_compress. Larger than PREFETCH_BUDGET_S because
# this runs once per compaction rather than once per turn, and compaction is already a
# multi-second LLM round-trip the user is waiting on — but still bounded, because it
# runs on the caller thread. Worst case: project detection (2s read) + the archive
# write (3s) + the recovery context (2s read, only started when it still fits).
PRE_COMPRESS_BUDGET_S = 7.0

# How many recently-ended session ids to remember, so an end followed by a switch
# cannot post /end twice. Bounded: a long-lived gateway process rotates sessions
# indefinitely, and an unbounded set would be a slow leak.
_ENDED_SESSION_MEMORY = 64

# Outcome of the compaction archive, which decides what the recovery text tells the
# model to do. Mirrors gentle-engram's ``ArchiveOutcome`` (compaction-recovery.js),
# minus its ``unavailable`` state, which covers a Pi-specific "could not identify the
# runtime session" case that our gates already turn into a plain ``""`` return.
_ARCHIVE_CONFIRMED = "confirmed"
_ARCHIVE_FAILED = "failed"
_ARCHIVE_UNKNOWN = "unknown"

# This provider's own tool names, whose results must never be passively captured.
# Upstream guards the same way (``if (ENGRAM_TOOL_NAMES.has(toolName.toLowerCase()))
# return;``, index.ts ``tool_execution_end``) and for a sharp reason: capturing the
# result of a `mem_search` would write recalled memories back in as new observations,
# a feedback loop that inflates the store every time the model looks something up.
# T6 owns the authoritative schema list in ``tools.py`` and must derive this set from
# it rather than keep a second copy.
_MEMORY_TOOL_NAMES = frozenset(
    {
        "mem_search",
        "mem_save",
        "mem_update",
        "mem_get_observation",
        "mem_context",
        "mem_session_summary",
        "mem_timeline",
    }
)

# Hermes core already gave its own background executor a 5s FIFO drain before it
# calls provider ``shutdown()``, and the user is waiting on process exit by then.
# Our queue holds at most a handful of small HTTP writes, each already bounded by
# the client's 3s write timeout, so a short join is the honest trade: finish what
# is in flight, abandon the rest rather than hold the CLI open.
SHUTDOWN_DRAIN_TIMEOUT_S = 2.0

# Bounded on purpose: with Engram down every job fails fast, but a wedged server
# that accepts connections and never answers would otherwise let the queue grow
# without limit for the life of the process.
_QUEUE_MAXSIZE = 256

# Static instructions for the model. A **module-level constant** so the string is
# byte-stable for the whole process: Hermes assembles the system prompt from
# provider blocks and the result is part of the cached prompt prefix, so a single
# dynamic value in here (a project name, an id, a count, a timestamp) would bust
# the provider's prompt cache on every turn it changed. Anything dynamic belongs in
# the per-turn recall block, which is injected into the user message instead.
SYSTEM_PROMPT_BLOCK = """## Engram memory (recall provider)

Relevant memory is recalled from Engram and injected automatically before every
non-trivial turn, so you already have the project's context — do not search just
to orient yourself.

### Saving (do this; do not wait to be asked)

Call `mem_save` immediately after: a bug fix, an architecture or design decision,
a non-obvious discovery about the codebase, a configuration or environment change,
a pattern or convention established, or a user preference or constraint you learned.

Shape of a save:
- **title**: verb + what — short and searchable
- **type**: bugfix | decision | architecture | discovery | pattern | config | preference
- **scope**: project (default) | personal | global
- **content**:
  **What**: one sentence — what was done
  **Why**: what motivated it
  **Where**: files or paths affected
  **Learned**: gotchas, edge cases, anything that surprised you

### Explicit recall

Use `mem_search` or `mem_context` only when the user asks you to recall past work,
or when the injected block is not enough — not to re-fetch what you already have.

### Closing

Before saying the work is done, call `mem_session_summary` with Goal, Discoveries,
Accomplished, Next Steps and Relevant Files."""


@dataclass(frozen=True)
class _Recall:
    """One rendered recall result, cached by :meth:`RecallProvider.queue_prefetch`."""

    query_key: str
    project: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    context: str = ""
    text: str = ""
    count: int = 0


class RecallProvider(MemoryProvider):
    """Engram-backed memory provider: deterministic recall before every turn.

    Fail-soft everywhere. Every hook swallows its own exceptions and degrades to
    "no memory this turn": the host calls these on the critical path of a turn, and
    a memory problem must never stop the agent from answering.
    """

    def __init__(
        self, config: RecallConfig | None = None, client: EngramClient | None = None
    ) -> None:
        """Construct the provider. No I/O, no threads, no network.

        Both collaborators are injectable for tests. Cheap construction is a hard
        requirement, not a preference: ``discover_memory_providers`` instantiates
        every candidate provider just to ask :meth:`is_available`, and the loader's
        fallback path constructs this class with no arguments at all.
        """
        self._config = config if config is not None else load_config()
        self._client = client if client is not None else EngramClient(self._config)

        # Session identity, filled by initialize().
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._platform: str = ""
        self._agent_context: str = "primary"
        self._writes_enabled: bool = True

        # Project detection cache.
        self._project: str = ""
        self._project_detected: bool = False

        # Recall state. Reset at the top of every prefetch so recall_status() can
        # only ever describe the last one.
        self._last_recall_count: int = 0
        self._last_recall_injected: bool = False
        self._prefetch_cache: _Recall | None = None
        self._context_fetched: bool = False

        # Capture state, per session.
        self._turn_number: int = 0
        # Content hash of the last compaction digest archived for this session, so a
        # compaction that fires twice over the same window writes once.
        self._last_pre_compress_hash: str = ""
        # Recently-ended session ids, used as a bounded ordered set (dicts preserve
        # insertion order) so /end is never posted twice for the same session.
        self._ended_sessions: dict[str, None] = {}
        # High-water mark into the host's message list: how many messages have
        # already been considered for passive capture. The host hands us the WHOLE
        # conversation every turn, so without this every turn would re-capture every
        # earlier turn's tool results.
        self._synced_message_count: int = 0

        # Background worker, started lazily on first use.
        self._queue: queue.Queue[tuple[str, Callable[[], None] | None]] = queue.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._shutting_down = False

    # -- Identity and availability -------------------------------------------

    @property
    def name(self) -> str:
        """The provider's name.

        Must stay ``"recall"``: the host resolves the active provider by matching
        ``memory.provider`` against the install-directory name / entry-point name,
        and both of those are ``recall`` (see ``plugin.yaml`` and ``pyproject.toml``).
        """
        return "recall"

    def is_available(self) -> bool:
        """Return whether the provider is configured and ready to activate.

        Config only — no network, no health check. Two reasons: the ABC documents
        it that way, and ``discover_memory_providers`` turns *any* raise in here
        into "unavailable", so a probe that threw would silently uninstall the
        feature. An Engram that is merely not running is still "available"; it just
        recalls nothing until it comes back.
        """
        try:
            if not self._config.enabled:
                return False
            return bool(str(self._config.url).strip())
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        """Explain an unavailable verdict for the loader's warning.

        Nothing in the Hermes tree observed so far actually calls this — the ABC
        documents it as the only place a provider can surface *why* it declined,
        since an unavailable provider is never initialized and so can never log.
        """
        try:
            if not self._config.enabled:
                return (
                    "recall is disabled by configuration; "
                    "set memory.recall.enabled: true in the Hermes config.yaml"
                )
            url = str(self._config.url).strip()
            return (
                "no Engram server URL resolved; start one with `engram serve` and point "
                f"memory.recall.url (or ENGRAM_URL) at it (resolved: {url or '<blank>'})"
            )
        except Exception:
            return "recall configuration could not be read"

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Record session identity and register the session with Engram in the background.

        Called once at agent startup, on the caller's thread, so this does no I/O
        inline: project detection and ``POST /sessions`` are both pushed onto the
        worker. Never raises — a memory backend that cannot be reached must not
        stop the agent from starting.
        """
        try:
            self._session_id = session_id or ""
            self._hermes_home = str(kwargs.get("hermes_home") or "")
            self._platform = str(kwargs.get("platform") or "")
            self._agent_context = str(kwargs.get("agent_context") or "primary")

            # Writes are allowed from the primary agent only. A cron run's system
            # prompt, a subagent's transcript or a flush pass would all land in the
            # user's project memory as if the user had said them — polluting the
            # store that every later recall reads from. Recall itself stays on for
            # every context (reading is harmless and useful); every write path in
            # this and later tasks must check this flag.
            self._writes_enabled = self._agent_context == "primary"

            if not self._config.enabled:
                return

            self._submit("register-session", self._register_session)
        except Exception:
            logger.debug("recall initialize failed (non-fatal)", exc_info=True)

    def _register_session(self) -> None:
        """Detect the project and register this session with Engram (worker thread)."""
        project = self._ensure_project()
        if not self._writes_enabled:
            logger.debug(
                "recall: agent_context=%s is not primary; skipping session registration",
                self._agent_context,
            )
            return
        if not self._session_id or not project:
            logger.debug(
                "recall: not registering session (session_id=%r project=%r)",
                self._session_id,
                project,
            )
            return
        self._client.create_session(self._session_id, project, directory=_working_directory())

    def shutdown(self) -> None:
        """Stop the worker: enqueue the sentinel and join briefly. Never raises.

        Safe when the worker never started and safe to call twice. The join is
        short by design (:data:`SHUTDOWN_DRAIN_TIMEOUT_S`) — see that constant.
        """
        try:
            with self._worker_lock:
                self._shutting_down = True
                thread = self._worker
                self._worker = None
            if thread is None:
                return
            try:
                self._queue.put_nowait(("shutdown", None))
            except queue.Full:
                logger.debug("recall: worker queue full at shutdown; abandoning pending jobs")
            thread.join(SHUTDOWN_DRAIN_TIMEOUT_S)
        except Exception:
            logger.debug("recall shutdown failed (non-fatal)", exc_info=True)

    # -- Background worker ---------------------------------------------------

    def _submit(self, label: str, fn: Callable[[], None]) -> None:
        """Enqueue a job for the worker thread, starting it on first use. Never raises.

        A full queue drops the job (debug log) rather than blocking the caller: the
        caller is a turn thread, and memory is never worth stalling a turn for.
        """
        try:
            with self._worker_lock:
                if self._shutting_down:
                    logger.debug("recall: dropping job %s, provider is shutting down", label)
                    return
                if self._worker is None or not self._worker.is_alive():
                    thread = threading.Thread(
                        target=self._worker_loop, name="recall-worker", daemon=True
                    )
                    self._worker = thread
                    thread.start()
            self._queue.put_nowait((label, fn))
        except queue.Full:
            logger.debug("recall: worker queue full; dropping job %s", label)
        except Exception:
            logger.debug("recall: could not enqueue job %s", label, exc_info=True)

    def _worker_loop(self) -> None:
        """Run queued jobs one at a time until the sentinel arrives.

        Serialized on purpose: Engram writes for a session must land in order, and
        one worker also caps thread growth at one per provider. Every job's
        exception is caught here, so a failing job can never kill the worker and
        silently take every later job with it.
        """
        while True:
            label, fn = self._queue.get()
            try:
                if fn is None:
                    return
                fn()
            except Exception:
                logger.debug("recall worker job %s failed (non-fatal)", label, exc_info=True)
            finally:
                self._queue.task_done()

    def _drain_for_test(self, timeout: float = 2.0) -> bool:
        """Test support only: block until the queue is empty and the in-flight job finished.

        Exists so async assertions do not need ``sleep`` calls. ``Queue.join()``
        takes no timeout, so it is joined through a throwaway thread. Returns
        ``True`` when the queue drained within ``timeout``.
        """
        waiter = threading.Thread(target=self._queue.join, daemon=True)
        waiter.start()
        waiter.join(timeout)
        return not waiter.is_alive()

    # -- Project detection ---------------------------------------------------

    def _detect_project(self) -> str:
        """Ask Engram which project the working directory belongs to. Never raises.

        ``TERMINAL_CWD`` wins over ``os.getcwd()``: the agent process may run from
        anywhere, and that variable is the user's actual terminal directory. An
        ambiguous project is a 200 with a blank ``project`` field — that is
        "unknown", not an error, and is reported as ``""`` just like a failure.
        """
        try:
            override = str(self._config.project_override or "").strip()
            if override:
                return override
            payload = self._client.current_project(_working_directory())
            if not isinstance(payload, dict):
                return ""
            project = payload.get("project")
            return project.strip() if isinstance(project, str) else ""
        except Exception:
            logger.debug("recall: project detection failed (non-fatal)", exc_info=True)
            return ""

    def _ensure_project(self) -> str:
        """Return the cached project, detecting it once on first use.

        Only a *successful* detection is memoised. A blank answer means either
        "Engram is down" or "the project is ambiguous", and both can resolve later
        in the session, so a later turn is allowed to ask again — the retry costs
        one request bounded by the client's 2s read timeout. The happy path pays
        for exactly one request per session.
        """
        if self._project_detected:
            return self._project
        project = self._detect_project()
        if project:
            self._project = project
            self._project_detected = True
        return project

    def _reset_project_cache(self) -> None:
        """Forget the detected project and any cached recall.

        The narrow reset: everything derived from *which project we are in*. Use
        :meth:`_reset_session_state` for a session switch, which also clears the
        per-session capture bookkeeping.
        """
        self._project = ""
        self._project_detected = False
        self._invalidate_prefetch_cache()
        self._context_fetched = False

    def _reset_session_state(self) -> None:
        """Forget everything scoped to one session.

        Which reset applies to which event:

        ======================  ============  ============  ===========  =========
        event                   project       context flag  turn caches  hash
        ======================  ============  ============  ===========  =========
        switch (reset=True)     re-detect     clear         clear        clear
        switch (reset=False)    re-detect     clear         clear        clear
        switch (rewound=True)   keep          keep          clear        keep
        ======================  ============  ============  ===========  =========

        A switch — ``/reset``, ``/new``, ``/resume``, ``/branch``, compression — can
        land in a different working directory, so the project is re-detected either
        way. A rewind keeps the same session and directory; only the per-turn caches
        are stale. ``_synced_message_count`` goes back to zero here because the new
        session starts a new transcript, and leaving it high would make the new
        session's early tool results look "already seen".
        """
        self._reset_project_cache()
        self._reset_turn_caches()
        self._turn_number = 0
        self._synced_message_count = 0
        self._last_pre_compress_hash = ""

    def _reset_turn_caches(self) -> None:
        """Drop state that describes one turn, keeping session and project state.

        This is the rewind case. ``_synced_message_count`` is deliberately **not**
        zeroed here: the transcript was truncated, but the observations captured from
        it are already in Engram, and re-capturing them would duplicate. The mark is
        instead clamped down to the list's real length by :meth:`_collect_captures`
        the next time a shorter list shows up.
        """
        self._invalidate_prefetch_cache()
        self._last_recall_count = 0
        self._last_recall_injected = False

    def _invalidate_prefetch_cache(self) -> None:
        """Drop the queued-prefetch cache. Exposed for T5's session-switch hook."""
        self._prefetch_cache = None

    # -- System prompt -------------------------------------------------------

    def system_prompt_block(self) -> str:
        """Return the static memory instructions, or ``""`` when recall is disabled.

        Returns the module constant verbatim; see :data:`SYSTEM_PROMPT_BLOCK` for
        why nothing dynamic may appear in it.
        """
        try:
            if not self._config.enabled:
                return ""
            return SYSTEM_PROMPT_BLOCK
        except Exception:
            return ""

    # -- Recall --------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall context for the upcoming turn as plain text. Never raises.

        Returns plain Markdown with **no** ``<memory-context>`` tags: Hermes wraps
        the result itself, and ``sanitize_context`` strips (and warns about) a
        provider that pre-wraps.

        The whole synchronous path runs under one wall-clock deadline
        (:data:`PREFETCH_BUDGET_S`), computed once here and threaded down, so a slow
        Engram degrades to *less* recall instead of overrunning the host's join.

        The recall state is cleared first, before any early return, so a skipped
        turn can never let :meth:`recall_status` report the previous turn's count.
        """
        self._last_recall_count = 0
        self._last_recall_injected = False
        try:
            if not self._config.enabled:
                return ""
            # The host gates trivial prompts too (agent/turn_context.py), but the
            # provider must not depend on that: anything calling prefetch()
            # directly still has to get the same answer.
            if is_trivial_prompt(query):
                return ""

            deadline = time.monotonic() + PREFETCH_BUDGET_S

            # Project resolution can itself cost a request: it is memoised only on a
            # successful detection, so a session whose first detection came back
            # blank asks again on a later turn.
            project = self._ensure_project()
            if not project:
                return ""

            query_key = _normalize_query(query)
            if not query_key:
                return ""

            cached = self._prefetch_cache
            if cached is not None and cached.query_key == query_key and cached.project == project:
                # Serving a cached answer costs no network time, so the budget never
                # blocks it: the deadline exists to bound requests, and there are
                # none on this path. Only the synchronous fetch below is gated, by
                # _build_recall, which abandons the search when the budget — already
                # possibly spent on project detection above — has run out.
                recall = cached
            else:
                recall = self._build_recall(query_key, project, deadline=deadline)

            if recall.text:
                self._last_recall_count = recall.count
                self._last_recall_injected = True
            logger.debug(
                "recall prefetch: project=%s session=%s hits=%d chars=%d",
                project,
                self._resolve_session_id(session_id),
                recall.count,
                len(recall.text),
            )
            return recall.text
        except Exception:
            logger.debug("recall prefetch failed (non-fatal)", exc_info=True)
            self._last_recall_count = 0
            self._last_recall_injected = False
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Precompute the next turn's recall in the background. Never raises.

        Hermes calls ``queue_prefetch_all`` on its own shared background worker, so
        this must return immediately and do the fetching on ours.

        The cache is single-entry and keyed on ``(normalised query, project)``, so a
        cached answer is only ever served for the exact query it was computed for —
        a stale recall is worse than none. It is invalidated on a session switch and
        whenever the project changes (:meth:`_reset_project_cache`).
        """
        try:
            if not self._config.enabled:
                return
            if is_trivial_prompt(query):
                return
            query_key = _normalize_query(query)
            if not query_key:
                return
            resolved_session = self._resolve_session_id(session_id)

            def job() -> None:
                project = self._ensure_project()
                if not project:
                    return
                # deadline=None: this runs on our worker, off the turn's critical
                # path, so it may take its time — and taking its time is the point,
                # since a background fetch that completes is what spares the next
                # turn a synchronous one. If it overruns, the next prefetch simply
                # finds no cache and fetches under its own budget.
                recall = self._build_recall(query_key, project, deadline=None)
                self._prefetch_cache = recall
                logger.debug(
                    "recall queue_prefetch: cached project=%s session=%s hits=%d",
                    project,
                    resolved_session,
                    recall.count,
                )

            self._submit("queue-prefetch", job)
        except Exception:
            logger.debug("recall queue_prefetch failed (non-fatal)", exc_info=True)

    def _build_recall(
        self, query_key: str, project: str, *, deadline: float | None = None
    ) -> _Recall:
        """Fetch and render one recall result. Runs on whichever thread asked for it.

        The per-turn **search runs first** and the once-per-session ``/context``
        second. That order is the whole point of the budget: the search is this
        turn's actual recall value, the project context is a one-off nice-to-have,
        so when time runs out the thing that gets dropped must be the optional one.

        ``deadline`` is a :func:`time.monotonic` instant, or ``None`` for "no limit"
        (the background path). When it has passed, the search is abandoned and an
        empty recall is returned; ``/context`` is held to a stricter rule — it only
        starts when the budget can still absorb a full read timeout.
        """
        if _deadline_passed(deadline):
            logger.debug("recall: budget spent before the search; returning no recall")
            return _Recall(query_key=query_key, project=project)

        hits = self._client.search(
            query=query_key,
            project=project,
            limit=self._config.recall_limit,
            match_mode="any",
        )

        context = ""
        # Once per *session*, not once per process: the flag is cleared by
        # _reset_project_cache() on a session switch.
        if not self._context_fetched:
            if _room_for_call(deadline):
                # Attempted. The flag is set before the call on purpose: a fetch
                # that was tried and FAILED must not be retried on every later turn
                # of the session.
                self._context_fetched = True
                context = self._client.context(project=project)
            else:
                # Never attempted — so it stays pending. This is the opposite
                # decision from a failed fetch above, and deliberately so: there is
                # no evidence the server cannot answer, only that this turn had no
                # time to ask. A later turn with budget to spare will pick it up.
                logger.debug("recall: no budget for the project context; leaving it pending")

        text, count = render_recall_block(hits, context, max_chars=self._config.recall_max_chars)
        return _Recall(
            query_key=query_key,
            project=project,
            hits=hits,
            context=context,
            text=text,
            count=count,
        )

    def recall_status(self) -> RecallStatus | None:
        """Describe what the last :meth:`prefetch` injected, or ``None`` when nothing was.

        Reflects only the most recent prefetch — the counters are cleared at the top
        of every call to it, so a skipped turn reports nothing rather than a stale
        count. ``count`` may be ``0`` when a block was injected without discrete
        hits (project context only); the host renders that as "recalled relevant
        memory".
        """
        try:
            if not self._last_recall_injected:
                return None
            label = f"Recall · {self._project}" if self._project else "Recall"
            return RecallStatus(provider_label=label, count=self._last_recall_count)
        except Exception:
            return None

    # -- Capture -------------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """Log the user's prompt to Engram. Runs on the turn thread; never blocks it.

        Trivial prompts are **not** skipped here, and that asymmetry with
        :meth:`prefetch` is deliberate: a prompt is the user's own words and the
        prompt log is a different artifact from recall. "go ahead please" is worth
        nothing as a search query and still worth having in the record of what was
        asked. The only floor is :data:`PROMPT_MIN_CHARS`, which is upstream's.
        """
        try:
            self._turn_number = turn_number if isinstance(turn_number, int) else 0

            if not self._write_gate_open():
                return

            content = self._prompt_content(message)
            if not content or len(content) <= PROMPT_MIN_CHARS:
                return

            session_id = self._session_id

            def job() -> None:
                project = self._ensure_project()
                if not project:
                    logger.debug("recall: no project; dropping prompt capture")
                    return
                self._client.add_prompt(session_id, content, project=project)

            self._submit("prompt-capture", job)
        except Exception:
            logger.debug("recall on_turn_start failed (non-fatal)", exc_info=True)

    def _prompt_content(self, message: str) -> str:
        """Prepare a user message for the prompt log: recover, redact, then cap.

        **Order note.** Upstream does ``stripPrivateTags(truncate(content, 2000))`` —
        truncate first. We redact first, on purpose. With upstream's order a
        ``<private>`` block that *straddles* the 2000-character boundary loses its
        closing tag to the cut, the non-greedy pattern then finds no match, and the
        opening fragment of the secret is stored. Redacting first removes complete
        blocks wherever they sit and only then caps the remainder, so the boundary
        can never turn a redaction into a leak. Nothing downstream can tell the
        difference except that the secret is gone, which is the point.
        """
        recovered = _user_instruction(message)
        if recovered is None:
            # A bare skill invocation carries no user instruction; there is nothing
            # the user actually said to log.
            logger.debug("recall: prompt is bare skill scaffolding; nothing to log")
            return ""
        return truncate(strip_private(recovered), PROMPT_MAX_CHARS)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Passively capture what the completed turn produced. Never raises.

        Mirrors gentle-engram's ``tool_execution_end``: the turn's new tool results
        and the assistant's reply are handed to Engram's extractor, which decides
        what is worth keeping.

        The host calls this on its own background worker, but the message list it
        passes is the agent's **live** list. The slice to capture is therefore
        snapshotted here, synchronously, before anything is queued — reading it later
        from our worker would race the next turn appending to it, or a compaction
        truncating it.

        ``messages`` is declared explicitly because the host probes the signature and
        only passes it when the parameter exists (``_provider_sync_accepts_messages``).
        """
        try:
            if not self._write_gate_open():
                return

            captures = self._collect_captures(assistant_content, messages)
            if not captures:
                return

            session_id_for_turn = self._resolve_session_id(session_id)
            if not session_id_for_turn:
                return

            def job() -> None:
                project = self._ensure_project()
                if not project:
                    logger.debug("recall: no project; dropping %d captures", len(captures))
                    return
                for source, content in captures:
                    # Guarded individually: one rejected capture must not abandon the
                    # rest of the turn's.
                    try:
                        self._client.passive_capture(
                            session_id_for_turn, content, project=project, source=source
                        )
                    except Exception:
                        logger.debug("recall: capture from %s failed", source, exc_info=True)

            self._submit("passive-capture", job)
        except Exception:
            logger.debug("recall sync_turn failed (non-fatal)", exc_info=True)

    def _collect_captures(
        self, assistant_content: str, messages: list[dict[str, Any]] | None
    ) -> list[tuple[str, str]]:
        """Build this turn's ``(source, content)`` captures and advance the high-water mark.

        The assistant's reply comes **first** so it survives
        :data:`PASSIVE_MAX_CAPTURES_PER_TURN`: it is the turn's own summary of what
        happened, and a turn with a dozen large tool results is exactly the turn
        whose summary is most worth keeping.
        """
        candidates: list[tuple[str, str]] = []

        if isinstance(assistant_content, str) and assistant_content:
            candidates.append(("assistant", assistant_content))

        if isinstance(messages, list):
            snapshot = list(messages)
            start = self._synced_message_count
            if start > len(snapshot):
                # The transcript shrank, so it was compacted or rewound under us.
                # Treat everything currently present as seen: re-posting a whole
                # history would be worse than missing one turn, and T5 owns the
                # explicit compaction/rewind hooks.
                logger.debug(
                    "recall: transcript shrank (%d -> %d); resetting capture mark",
                    start,
                    len(snapshot),
                )
                start = len(snapshot)
            candidates.extend(_extract_tool_captures(snapshot, start))
            self._synced_message_count = len(snapshot)

        captures: list[tuple[str, str]] = []
        for source, raw in candidates:
            # Upstream measures the floor on the RAW content, before redaction, so a
            # long block that redacts down to almost nothing is still offered up.
            if len(raw) <= PASSIVE_MIN_CHARS:
                continue
            content = strip_private(raw)
            if not content:
                continue
            if len(captures) >= PASSIVE_MAX_CAPTURES_PER_TURN:
                logger.debug(
                    "recall: hit the per-turn capture cap (%d); dropping the rest",
                    PASSIVE_MAX_CAPTURES_PER_TURN,
                )
                break
            captures.append((source, content))
        return captures

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
    ) -> None:
        """Capture a completed subagent delegation as one observation. Never raises.

        ``child_session_id`` is logged but kept **out** of the captured content: the
        subagent runs with no provider session of its own (the ABC notes subagents are
        created with ``skip_memory=True``), so the id refers to nothing retrievable
        from Engram and would be an unresolvable reference inside the observation.
        """
        try:
            if not self._write_gate_open():
                return

            composed = _compose_delegation(task, result)
            if len(composed) <= PASSIVE_MIN_CHARS:
                return
            content = strip_private(composed)
            if not content:
                return

            session_id = self._session_id
            logger.debug("recall: capturing delegation from child session %r", child_session_id)

            def job() -> None:
                project = self._ensure_project()
                if not project:
                    logger.debug("recall: no project; dropping delegation capture")
                    return
                self._client.passive_capture(
                    session_id, content, project=project, source="subagent"
                )

            self._submit("delegation-capture", job)
        except Exception:
            logger.debug("recall on_delegation failed (non-fatal)", exc_info=True)

    def _write_gate_open(self) -> bool:
        """Whether this provider may write to Engram at all right now.

        One gate for every capture path, so "disabled", "not the primary agent" and
        "no registered session" can never be enforced in three slightly different
        ways. The project check is deliberately *not* here: it can cost a request, so
        it belongs on the worker.
        """
        if not self._config.enabled:
            return False
        if not self._writes_enabled:
            # Non-primary agent_context (subagent, cron, flush). A cron system prompt
            # or a subagent transcript would land in the user's project memory as if
            # the user had said it.
            logger.debug("recall: writes disabled for agent_context=%s", self._agent_context)
            return False
        return bool(self._session_id)

    # -- Session lifecycle ---------------------------------------------------

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Close the session's record in Engram. Never raises.

        Sends an **empty** summary, deliberately. Upstream gentle-engram does the
        same — its ``/sessions/{id}/end`` body is ``{"summary": ""}`` and it never
        synthesises one — because the meaningful end-of-session artifact is the
        model-authored ``mem_session_summary``, written while the model still has the
        conversation in context. A provider-side digest of the transcript would be a
        second, worse summary competing with it for the same topic.

        ``messages`` is therefore unused: closing a session needs no transcript, and
        the one place we *do* digest a transcript is :meth:`on_pre_compress`, where
        the transcript is about to be destroyed and nothing else will preserve it.
        """
        try:
            if not self._write_gate_open():
                return

            session_id = self._session_id
            # Recorded synchronously, not from the worker: the host submits
            # on_session_end and on_session_switch as one background task, back to
            # back, so the switch's de-duplication check can run before our worker
            # has picked the end job up.
            self._remember_ended_session(session_id)

            def job() -> None:
                self._client.end_session(session_id, summary="")

            self._submit("session-end", job)
        except Exception:
            logger.debug("recall on_session_end failed (non-fatal)", exc_info=True)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Re-bind every piece of per-session state to the new session id. Never raises.

        Fires on ``/resume``, ``/branch``, ``/reset``, ``/new``, their gateway
        equivalents and context compression. ``**kwargs`` absorbs what the host adds
        per path (it forwards ``reason`` on the rotation path, and ``rewound`` only
        when true), so nothing here depends on an exact kwargs shape.
        """
        try:
            previous_session_id = self._session_id
            unchanged = bool(new_session_id) and new_session_id == previous_session_id

            if rewound and unchanged:
                # The transcript was truncated but the session and directory are the
                # same: the record stays open, nothing is re-registered, and only the
                # per-turn caches are stale.
                logger.debug("recall: session %s rewound; dropping turn caches", new_session_id)
                self._reset_turn_caches()
                return

            if reset:
                self._end_outgoing_session(previous_session_id, new_session_id)

            self._session_id = new_session_id or ""
            self._reset_session_state()

            if self._session_id:
                logger.debug(
                    "recall: switched to session %s (reset=%s, parent=%r)",
                    self._session_id,
                    reset,
                    parent_session_id,
                )
                # Same code path initialize() uses, so registration cannot drift
                # between "first session" and "every session after it".
                self._submit("register-session", self._register_session)
        except Exception:
            logger.debug("recall on_session_switch failed (non-fatal)", exc_info=True)

    def _end_outgoing_session(self, previous_session_id: str, new_session_id: str) -> None:
        """End the session being left behind, at most once.

        ``reset=True`` means a genuinely new conversation, so the old one is over and
        its record should be closed rather than left open forever.

        The de-duplication matters because of the host's ordering: on the rotation
        path ``MemoryManager`` submits ``on_session_end`` and then
        ``on_session_switch(..., reset=True)`` as a single background task, in that
        exact order. Without the check, every ``/reset`` would post ``/end`` twice for
        the same session.
        """
        if not previous_session_id or previous_session_id == new_session_id:
            return
        if not self._writes_enabled or not self._config.enabled:
            return
        if previous_session_id in self._ended_sessions:
            logger.debug("recall: session %s already ended; not ending twice", previous_session_id)
            return

        self._remember_ended_session(previous_session_id)

        def job() -> None:
            self._client.end_session(previous_session_id, summary="")

        self._submit("session-end", job)

    def _remember_ended_session(self, session_id: str) -> None:
        """Note that a session's ``/end`` has been issued, keeping the set bounded."""
        if not session_id:
            return
        self._ended_sessions[session_id] = None
        while len(self._ended_sessions) > _ENDED_SESSION_MEMORY:
            self._ended_sessions.pop(next(iter(self._ended_sessions)))

    # -- Compaction ----------------------------------------------------------

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Archive a digest of the window about to be discarded, and return recovery text.

        Two jobs, in order: persist something that survives the compaction, then tell
        the model what it needs to know about what just happened.

        **The archive write is synchronous**, unlike every other write in this
        provider. The reason is the return value: the recovery text has to tell the
        model either "it is already saved, do nothing" or "save it yourself now", and
        a queued write makes both of those claims unprovable — the honest answer would
        always be "I cannot tell, go and check", which costs the model a verification
        search after every single compaction. Doing the write here, inside a bounded
        budget (:data:`PRE_COMPRESS_BUDGET_S`), buys a truthful answer. Compaction is
        rare and already a multi-second LLM round-trip, so the cost lands where there
        is room for it.

        Returns ``""`` when there is nothing we can contribute, which for a v1
        provider is the correct degraded behaviour: the host simply joins nothing.
        """
        try:
            if not self._write_gate_open():
                return ""

            deadline = time.monotonic() + PRE_COMPRESS_BUDGET_S
            session_id = self._session_id

            project = self._ensure_project()
            if not project:
                logger.debug("recall: no project at compaction; contributing nothing")
                return ""

            outcome = self._archive_compaction_digest(session_id, project, messages, deadline)
            context = self._compaction_context(session_id, deadline)
            return _build_recovery_notice(project, context, outcome)
        except Exception:
            logger.debug("recall on_pre_compress failed (non-fatal)", exc_info=True)
            return ""

    def _archive_compaction_digest(
        self,
        session_id: str,
        project: str,
        messages: list[dict[str, Any]] | None,
        deadline: float | None,
    ) -> str:
        """Post the compaction digest and report how it went.

        Idempotent by content hash: compaction can fire more than once over the same
        window, and a second identical ``session_summary`` is noise in the user's
        store. A repeat is reported as ``confirmed`` because the content *is* already
        archived — which is exactly what the model needs to know.
        """
        content = _compose_compaction_digest(messages)
        if not content:
            return _ARCHIVE_FAILED

        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest == self._last_pre_compress_hash:
            logger.debug("recall: compaction digest unchanged; already archived")
            return _ARCHIVE_CONFIRMED

        if not _room_for_call(deadline, client_module.WRITE_TIMEOUT_S):
            # Never attempted, so nothing was written — but say "unknown" rather than
            # "failed": the model's safe move is to verify before saving, and claiming
            # a definite failure would invite a duplicate if an earlier compaction had
            # in fact archived something.
            logger.debug("recall: no budget to archive the compaction digest")
            return _ARCHIVE_UNKNOWN

        saved = self._client.add_observation(
            session_id,
            "Compaction recovery summary",
            content,
            type_="session_summary",
            project=project,
            scope="project",
            topic_key="session/compaction-recovery",
        )
        if isinstance(saved, dict):
            self._last_pre_compress_hash = digest
            return _ARCHIVE_CONFIRMED

        # A write that timed out may still have been committed server-side, so it is
        # genuinely ambiguous; anything else (refused, rejected, 4xx/5xx) definitely
        # was not written.
        if client_module.TIMEOUT_ERROR_MARKER in self._client.last_error_text():
            return _ARCHIVE_UNKNOWN
        return _ARCHIVE_FAILED

    def _compaction_context(self, session_id: str, deadline: float | None) -> str:
        """Fetch this session's compaction context, capped; ``""`` when unavailable."""
        if not _room_for_call(deadline):
            logger.debug("recall: no budget for the compaction context")
            return ""
        return truncate(self._client.compaction_context(session_id), RECOVERY_MAX_CHARS)

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return this provider's tool schemas — empty until T6 fills it from ``tools.py``.

        Cheap, pure and side-effect free: the host calls it at least three times
        during registration alone (conflict check, routing table, log line).
        """
        return []

    # -- Internals -----------------------------------------------------------

    def _resolve_session_id(self, session_id: str) -> str:
        """Return the session id to act on: the argument when non-blank, else the stored one.

        ``MemoryManager.prefetch_all`` calls ``provider.prefetch(query, session_id=...)``
        but is itself invoked positionally with no session id, so the keyword
        arrives as ``""``. The id registered in :meth:`initialize` is therefore the
        real one in the CLI case, while a gateway serving concurrent sessions does
        pass an explicit id — which wins when it is there.
        """
        candidate = session_id.strip() if isinstance(session_id, str) else ""
        return candidate or self._session_id


def _normalize_query(query: str) -> str:
    """Strip the turn text and cap it at :data:`QUERY_MAX_CHARS`."""
    if not isinstance(query, str):
        return ""
    return query.strip()[:QUERY_MAX_CHARS]


def _deadline_passed(deadline: float | None) -> bool:
    """Whether the wall-clock budget is spent. ``None`` means "no deadline"."""
    return deadline is not None and time.monotonic() >= deadline


def _room_for_call(deadline: float | None, timeout_s: float | None = None) -> bool:
    """Whether a *skippable* client call still fits inside the budget.

    Stricter than :func:`_deadline_passed`: it demands enough remaining budget for a
    whole timeout, so an optional call can never be the reason the total overruns the
    deadline. "The deadline has not passed yet" is not enough for that — a call started
    one millisecond before it can still run for the full timeout, and the sum would be
    ``budget + timeout``, which is the bound a budget exists to remove.

    ``timeout_s`` defaults to the client's read timeout; a write costs more, so the
    caller passes ``WRITE_TIMEOUT_S``. Both are read through the client module
    namespace at call time, not bound at import, so a test that monkeypatches them is
    actually honoured.
    """
    if deadline is None:
        return True
    budget = client_module.READ_TIMEOUT_S if timeout_s is None else timeout_s
    return (deadline - time.monotonic()) >= budget


def _working_directory() -> str:
    """The directory to resolve the project from: ``TERMINAL_CWD`` when set, else the cwd."""
    terminal_cwd = os.environ.get("TERMINAL_CWD", "")
    if terminal_cwd.strip():
        return terminal_cwd.strip()
    try:
        return os.getcwd()
    except OSError:
        return ""


def _extract_tool_captures(messages: list[Any], start: int) -> list[tuple[str, str]]:
    """Return ``(source, content)`` for every tool result at or after index ``start``.

    A tool result is a message with ``role == "tool"`` and non-empty string content.
    The source is the message's own ``name``, else the function name of the matching
    ``tool_call_id`` on an earlier assistant message, else the literal ``"tool"``.

    Anything whose shape is not recognised is **skipped, not guessed**: this parses a
    transcript produced by someone else, and inventing a capture out of a shape we do
    not understand is how a memory store fills with nonsense. Results from this
    provider's own tools are skipped too — see :data:`_MEMORY_TOOL_NAMES`.
    """
    tool_call_names = _tool_call_names(messages)
    captures: list[tuple[str, str]] = []
    for message in messages[max(0, start) :]:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        source = _tool_source(message, tool_call_names)
        if source.lower() in _MEMORY_TOOL_NAMES:
            continue
        captures.append((source, content))
    return captures


def _tool_call_names(messages: list[Any]) -> dict[str, str]:
    """Map ``tool_call_id`` to function name, from every assistant message present.

    Built over the whole snapshot rather than the new slice: the assistant message
    that issued a call is normally in the same turn, but nothing guarantees it, and a
    missing entry only costs the generic ``"tool"`` source.
    """
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function = call.get("function")
            if not isinstance(call_id, str) or not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                names[call_id] = name.strip()
    return names


def _tool_source(message: dict[str, Any], tool_call_names: dict[str, str]) -> str:
    """Best available name for the tool that produced this result."""
    name = message.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    call_id = message.get("tool_call_id")
    if isinstance(call_id, str):
        resolved = tool_call_names.get(call_id)
        if resolved:
            return resolved
    return "tool"


def _compose_delegation(task: str, result: str) -> str:
    """Render a delegation as one observation body, each half capped independently."""
    task_text = truncate(task if isinstance(task, str) else "", DELEGATION_TASK_MAX_CHARS)
    result_text = truncate(result if isinstance(result, str) else "", DELEGATION_RESULT_MAX_CHARS)
    return f"**Delegated task**: {task_text}\n\n**Subagent result**: {result_text}"


def _compose_compaction_digest(messages: list[dict[str, Any]] | None) -> str:
    """Summarise the window about to be discarded, bounded and redacted.

    A digest, not the transcript: the transcript is what compaction exists to shed, and
    re-posting it whole would just move the problem into the user's memory store. What
    survives is the shape of the conversation — how long it ran, what the user asked,
    and where the assistant left off — which is what someone resuming needs.
    """
    rows = [m for m in (messages or []) if isinstance(m, dict)]

    user_turns = [
        text
        for text in (_collapse(r.get("content")) for r in rows if r.get("role") == "user")
        if text
    ]
    assistant_replies = [
        text
        for text in (_collapse(r.get("content")) for r in rows if r.get("role") == "assistant")
        if text
    ]

    if not user_turns and not assistant_replies:
        return ""

    lines = [
        "**What**: transcript window compacted before it was discarded.",
        f"**Messages**: {len(rows)}",
    ]
    if user_turns:
        lines.append("**User asked**:")
        for turn in user_turns[-_DIGEST_USER_TURNS:]:
            lines.append(f"- {truncate(turn, _DIGEST_USER_CHARS)}")
    if assistant_replies:
        last = truncate(assistant_replies[-1], _DIGEST_ASSISTANT_CHARS)
        lines.append(f"**Last assistant reply**: {last}")

    return truncate(strip_private("\n".join(lines)), PRE_COMPRESS_MAX_CHARS)


def _collapse(content: Any) -> str:
    """Collapse a message's text content to one whitespace-normalised line, else ``""``."""
    if not isinstance(content, str):
        return ""
    return " ".join(content.split())


def _build_recovery_notice(project: str, context: str, outcome: str) -> str:
    """Compose the text handed back to the compression summary prompt.

    The instruction is chosen by archive outcome, adapted from gentle-engram's
    ``buildRecoveryNotice`` (compaction-recovery.js) — the same three decisions,
    reworded rather than copied, since the upstream strings name gentle-engram and Pi's
    own tooling.
    """
    if outcome == _ARCHIVE_CONFIRMED:
        instruction = (
            f"A compaction recovery summary for '{project}' is already saved in Engram. "
            "Do NOT call mem_session_summary for this compaction — it would duplicate "
            "the record. Call mem_context if you need more recent project memory."
        )
    elif outcome == _ARCHIVE_UNKNOWN:
        instruction = (
            f"Whether the compaction recovery summary reached Engram ('{project}') could "
            "not be confirmed. Do NOT retry or call mem_session_summary yet, because "
            "that could duplicate the summary. Verify with mem_search first, and save it "
            "only if it is genuinely absent."
        )
    else:
        instruction = (
            "CRITICAL INSTRUCTION FOR COMPACTED SUMMARY:\n"
            f"The compaction recovery summary was NOT saved to Engram ('{project}'). "
            "FIRST ACTION REQUIRED: call mem_session_summary with the content of this "
            "compacted summary, before any other work."
        )
    trimmed = context.strip() if isinstance(context, str) else ""
    return f"{trimmed}\n\n{instruction}" if trimmed else instruction


def _user_instruction(message: Any) -> str | None:
    """Recover the user's own words from a possibly skill-expanded turn message.

    ``None`` means "this turn carried no user instruction" (a bare ``/skill``
    invocation), which the caller treats as nothing to log.

    Why this exists: when the user invokes a ``/skill`` or ``/bundle``, Hermes expands
    the turn into a model-facing message embedding the entire skill body.
    ``MemoryManager`` strips that for ``prefetch_all`` and ``sync_all`` — its own
    docstring says feeding it verbatim to providers "pollutes their stores/embeddings
    with prompt scaffolding instead of what the user actually asked" — but the
    ``on_turn_start`` fan-out does **not** strip it, so the raw expansion reaches us.
    Rather than guess the scaffolding format, borrow the host's own extractor when it
    is importable and otherwise pass the message through unchanged.
    """
    if not isinstance(message, str):
        return ""
    extractor = _skill_instruction_extractor()
    if extractor is None:
        return message
    try:
        recovered = extractor(message)
    except Exception:
        return message
    if recovered is None:
        return None
    return recovered if isinstance(recovered, str) else message


_EXTRACTOR_UNRESOLVED = object()
_skill_extractor: Any = _EXTRACTOR_UNRESOLVED


def _skill_instruction_extractor() -> Any:
    """Resolve Hermes' skill-message extractor once, caching the miss as well as the hit.

    Caching the *failure* matters: Python does not cache a failed import, so retrying
    would re-scan ``sys.path`` on every single turn — on the prompt path, which runs
    before prefetch. Outside Hermes (every test run) the answer is ``None``.
    """
    global _skill_extractor
    if _skill_extractor is _EXTRACTOR_UNRESOLVED:
        try:
            from agent.skill_commands import (  # type: ignore[import-not-found]
                extract_user_instruction_from_skill_message,
            )

            _skill_extractor = extract_user_instruction_from_skill_message
        except Exception:
            _skill_extractor = None
    return _skill_extractor
