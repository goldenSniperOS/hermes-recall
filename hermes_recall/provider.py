"""``RecallProvider``: the Hermes ``MemoryProvider`` hooks (recall, capture, lifecycle)."""

from __future__ import annotations

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

        T5 calls this from ``on_session_switch``: a switch can follow a directory
        change, so the project has to be re-detected and a recall computed for the
        old project must not be served for the new one.
        """
        self._project = ""
        self._project_detected = False
        self._invalidate_prefetch_cache()
        self._context_fetched = False

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


def _room_for_call(deadline: float | None) -> bool:
    """Whether a *skippable* client call still fits inside the budget.

    Stricter than :func:`_deadline_passed`: it demands enough remaining budget for a
    whole read timeout, so an optional call can never be the reason the total
    overruns the deadline. "The deadline has not passed yet" is not enough for that —
    a call started one millisecond before it can still run for the full timeout, and
    the sum would be ``PREFETCH_BUDGET_S + READ_TIMEOUT_S``, which is the bound this
    budget exists to remove.

    ``READ_TIMEOUT_S`` is read through the client module namespace at call time, not
    bound at import, so a test that monkeypatches it is actually honoured.
    """
    if deadline is None:
        return True
    return (deadline - time.monotonic()) >= client_module.READ_TIMEOUT_S


def _working_directory() -> str:
    """The directory to resolve the project from: ``TERMINAL_CWD`` when set, else the cwd."""
    terminal_cwd = os.environ.get("TERMINAL_CWD", "")
    if terminal_cwd.strip():
        return terminal_cwd.strip()
    try:
        return os.getcwd()
    except OSError:
        return ""
