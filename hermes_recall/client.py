"""Stdlib HTTP client for the local Engram server: project detection, search, writes.

Every public method is fail-soft: no exception escapes, failures return the
documented empty value (``None``, ``[]`` or ``""``) and leave a human-readable
reason in :meth:`EngramClient.last_error_text`. The provider sits on the critical
path of every Hermes turn, so an unreachable or broken Engram must degrade to
"no memory this turn", never to a traceback.

Endpoint shapes are verified against a live Engram 2.0.0 server and the route
table in ``engram/server.go``.
"""

from __future__ import annotations

import errno
import json
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .config import RecallConfig, load_config

logger = logging.getLogger(__name__)

# Timeouts are deliberately far below the host's patience. Hermes joins an
# external provider's prefetch for `_EXTERNAL_PREFETCH_TIMEOUT_S = 8.0` seconds
# (agent/memory_manager.py); on timeout `_prefetch_provider` leaves the thread
# running and registers it, and every later turn that finds it still alive logs
# "prefetch is still running; skipping this turn" and returns "" immediately.
# One stuck HTTP call therefore disables recall for the rest of the session, not
# just for the turn that blocked. So the client must give up long before 8s.
READ_TIMEOUT_S = 2.0
WRITE_TIMEOUT_S = 3.0

# Marker the recorded reason carries when a request was abandoned on a timeout, as
# opposed to refused or rejected. The distinction is not cosmetic: a POST that timed
# out may still have been committed server-side, so a caller deciding whether to tell
# the model "save it yourself" has to tell "definitely not written" apart from "might
# have been written". Every method collapses failures to the same empty value, so this
# is the only channel that carries it. Upstream gentle-engram draws the same line with
# ``result.transportFailure?.outcome === "unknown"``.
TIMEOUT_ERROR_MARKER = "timed out"

# Keys `PATCH /observations/{id}` accepts, mapped to the client kwarg name.
_UPDATABLE_FIELDS = ("type", "title", "content", "find", "replace", "project", "scope", "topic_key")

# The only `match_mode` values the server accepts; anything else is a 400.
_MATCH_MODES = frozenset({"all", "any"})

# Bounds `POST /conflicts/judge-many` enforces on its batch (agentops.JudgmentBatch).
JUDGE_MANY_MAX = 32


class EngramClient:
    """Thin, fail-soft HTTP client for one Engram server."""

    def __init__(self, config: RecallConfig | None = None) -> None:
        self._config = config if config is not None else load_config()
        # `load_config` already normalises the URL, but a caller can hand in a
        # RecallConfig it built itself. Re-stripping here makes "no trailing
        # slash" an invariant of the client rather than a promise of its input:
        # a trailing slash would otherwise produce `//health`, which Go's
        # ServeMux answers with a redirect instead of the route.
        self._base_url = self._config.url.strip().rstrip("/")
        # Per thread: the provider's background worker and the tool-call thread
        # share one client, and a failure on one must not be reported (or erased)
        # as the other's. Every caller reads the reason on the thread that made
        # the call.
        self._errors = threading.local()
        self._opener = _build_opener(self._base_url)
        # Called (no arguments) whenever a request is refused at the TCP level: the
        # provider's one bounded restart trigger. ``None`` means nobody listens.
        self._refusal_listener: Callable[[], None] | None = None

    def set_refusal_listener(self, listener: Callable[[], None] | None) -> None:
        """Register the callback fired when the server refuses a connection.

        Only a refusal counts — proof that nothing listens on the port. A timeout or
        an HTTP error means a server answered (or might have), and restarting over
        either could fight a live server. The callback runs on the requesting
        thread, so it must be cheap; its exceptions are swallowed.
        """
        self._refusal_listener = listener

    # -- Introspection -------------------------------------------------------

    @property
    def base_url(self) -> str:
        """The server's base URL, normalised with no trailing slash."""
        return self._base_url

    @property
    def last_error(self) -> str:
        """The most recent failure reason on this thread, or ``""`` after a success."""
        return getattr(self._errors, "message", "")

    def last_error_text(self) -> str:
        """Return the last failure reason on this thread, or ``""`` when healthy."""
        return getattr(self._errors, "message", "")

    @property
    def last_error_status(self) -> int | None:
        """HTTP status of the last failure on this thread, when a server answered it.

        ``None`` after a success, and after a failure no server answered: refused,
        unreachable, timed out, or refused locally before any request. The structured
        twin of :meth:`last_error_text`, so a caller can tell "Engram said no" from
        "Engram is not there" without parsing the reason.
        """
        return getattr(self._errors, "status", None)

    @property
    def last_error_body(self) -> dict[str, Any] | None:
        """The JSON object of the last HTTP error answer on this thread, if it had one.

        Engram 3.6 refusals carry a machine-readable ``code`` and context (for example
        ``available_projects``, or a consolidation's fresh ``plan``) next to ``error``.
        ``None`` after a success, after a failure no server answered, and when the
        error body was not a JSON object (an older server's unknown route answers a
        plain-text 404).
        """
        return getattr(self._errors, "body", None)

    # -- Transport -----------------------------------------------------------

    def _record_error(
        self, message: str, status: int | None = None, body: dict[str, Any] | None = None
    ) -> None:
        """Store a failure reason (and its HTTP status and body, if any); log at debug only.

        Never above debug: the provider is fail-soft, and an Engram server that
        is simply not running would otherwise print a warning on every turn.
        """
        self._errors.message = message
        self._errors.status = status
        self._errors.body = body
        logger.debug("engram client: %s", message)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Perform one HTTP request and return the decoded JSON, or ``None`` on failure.

        Query params whose value is ``None`` or ``""`` are dropped — the server
        rejects a blank ``project`` outright, and an omitted param is how its
        defaults are selected. ``0`` and ``False`` are real values and survive as
        ``"0"`` / ``"false"``.
        """
        url = f"{self._base_url}{path}"
        if params:
            filtered = {
                key: _query_value(value)
                for key, value in params.items()
                if value is not None and value != ""
            }
            if filtered:
                url = f"{url}?{urllib.parse.urlencode(filtered)}"

        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self._config.auth_token:
            headers["Authorization"] = f"Bearer {self._config.auth_token}"

        if timeout is None:
            # Read through the module namespace rather than binding the constants
            # as default argument values: defaults are evaluated once at function
            # definition, so a test that monkeypatches READ_TIMEOUT_S/
            # WRITE_TIMEOUT_S would otherwise have no effect.
            timeout = READ_TIMEOUT_S if method == "GET" else WRITE_TIMEOUT_S

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read()
            self._errors.message = ""
            self._errors.status = None
            self._errors.body = None
            if not raw:
                return None
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # A 2xx that is not JSON is not an exception to propagate; the
                # caller's documented empty value is the right answer.
                self._record_error(f"{method} {path}: response body was not valid JSON")
                return None
        except urllib.error.HTTPError as exc:
            # Never include the request body in the recorded reason: it carries
            # user content (observation text, prompts, search queries).
            payload = _http_error_payload(exc)
            detail = _http_error_detail(exc, payload)
            self._record_error(f"{method} {path}: HTTP {exc.code}{detail}", exc.code, payload)
            return None
        except urllib.error.URLError as exc:
            self._record_error(f"{method} {path}: unreachable ({exc.reason})")
            if _is_connection_refused(exc.reason):
                self._notify_refused()
            return None
        except TimeoutError:
            self._record_error(f"{method} {path}: {TIMEOUT_ERROR_MARKER} after {timeout:g}s")
            return None
        except OSError as exc:
            self._record_error(f"{method} {path}: OS error ({exc})")
            if _is_connection_refused(exc):
                self._notify_refused()
            return None
        except json.JSONDecodeError as exc:
            self._record_error(f"{method} {path}: malformed JSON ({exc.msg})")
            return None
        except Exception as exc:  # noqa: BLE001 - fail-soft backstop
            self._record_error(f"{method} {path}: unexpected {type(exc).__name__}")
            return None

    def _notify_refused(self) -> None:
        """Fire the refusal listener, never letting it raise into a request."""
        listener = self._refusal_listener
        if listener is None:
            return
        try:
            listener()
        except Exception:
            logger.debug("engram client: refusal listener failed", exc_info=True)

    # -- Health and project --------------------------------------------------

    def health(self, timeout: float | None = None) -> dict[str, Any] | None:
        """``GET /health`` → ``{status, service, version, instance_id}``.

        ``timeout`` overrides the read timeout; the auto-start readiness poll uses a
        short one so a deadline is spent on attempts rather than one hung probe.
        """
        return _as_dict(self._request("GET", "/health", timeout=timeout))

    def current_project(self, cwd: str) -> dict[str, Any] | None:
        """``GET /project/current`` → the project Engram resolves for ``cwd``.

        Returns ``{project, project_source, project_path, cwd, available_projects}``.
        A 200 whose ``project`` is ``""`` and which carries an ``error_hint`` is
        Engram's *ambiguous project* answer, not a failure — it is returned
        unchanged so the caller can ask the user to choose.
        """
        return _as_dict(self._request("GET", "/project/current", params={"cwd": cwd}))

    # -- Sessions ------------------------------------------------------------

    def create_session(
        self, session_id: str, project: str, directory: str = ""
    ) -> dict[str, Any] | None:
        """``POST /sessions`` → ``{"id", "status"}`` (HTTP 201).

        Always sent with ``"resume": true``, as gentle-engram 0.2.0 does (Engram v3,
        6af3a4b): when ``session_id`` already ended, the server picks the lowest live
        ``<id>:resume:<N>`` continuation (or the next ordinal) and answers with that
        ``id`` plus ``"resumed_from"``. Without it v2.2+ answers 409
        ``session_already_ended``; Engram 2.0.0 ignores the field and echoes the id.
        The ack is returned unchanged — validating the selected id is the caller's job,
        since only the caller knows which runtime session it registered for.
        """
        body: dict[str, Any] = {
            "id": session_id,
            "project": project,
            "ownership_mode": "project_owned",
            "resume": True,
        }
        if directory.strip():
            body["directory"] = directory
        return _as_dict(self._request("POST", "/sessions", body=body))

    def end_session(self, session_id: str, summary: str = "") -> dict[str, Any] | None:
        """``POST /sessions/{id}/end`` → ``{"id", "status"}``."""
        quoted = urllib.parse.quote(session_id, safe="")
        return _as_dict(self._request("POST", f"/sessions/{quoted}/end", body={"summary": summary}))

    # -- Observations --------------------------------------------------------

    def add_observation(
        self,
        session_id: str,
        title: str,
        content: str,
        *,
        type_: str = "manual",
        project: str = "",
        scope: str = "project",
        topic_key: str = "",
        tool_name: str = "",
    ) -> dict[str, Any] | None:
        """``POST /observations`` → ``{"id": <int>, "status": "saved"}`` (HTTP 201)."""
        # The server requires session_id, title and content, and validates the
        # title first (a whitespace-only title is rejected). Checking here turns
        # a guaranteed 400 into a free local answer.
        missing = [
            name
            for name, value in (
                ("session_id", session_id),
                ("title", title),
                ("content", content),
            )
            if not value.strip()
        ]
        if missing:
            self._record_error(
                "POST /observations: refused locally, blank required field(s): "
                + ", ".join(missing)
            )
            return None

        body: dict[str, Any] = {
            "session_id": session_id,
            "type": type_,
            "title": title,
            "content": content,
        }
        for key, value in (
            ("project", project),
            ("scope", scope),
            ("topic_key", topic_key),
            ("tool_name", tool_name),
        ):
            if value.strip():
                body[key] = value
        return _as_dict(self._request("POST", "/observations", body=body))

    def update_observation(
        self, observation_id: int, *, expected_project: str = "", **fields: Any
    ) -> dict[str, Any] | None:
        """``PATCH /observations/{id}?expected_project=`` → the full updated observation.

        Only the keys explicitly passed are sent. The server's patch params are
        **pointers** (``store.UpdateObservationParams``), so an absent key and an
        empty-string key are different instructions: absent means "leave it
        alone", empty means "set it to empty". Dropping ``None`` preserves that
        distinction while giving callers a way to say "not this one"; unknown
        keys are ignored rather than forwarded.

        ``expected_project`` is the caller's assertion of who owns the observation.
        Engram v3 (e8c0ce3) requires it as a query param — 400 when missing, 409
        when it does not match the owner — and treats it as an explicit claim, never
        a default, so it is not defaulted here either. Engram 2.0.0 through 2.2.1 never
        read the query string on this route, so there it has no effect.
        """
        body = {
            key: value
            for key, value in fields.items()
            if key in _UPDATABLE_FIELDS and value is not None
        }
        if not body:
            # The server 400s on an empty patch ("at least one field is required").
            self._record_error(
                f"PATCH /observations/{observation_id}: refused locally, no updatable fields given"
            )
            return None
        expected = expected_project.strip() if isinstance(expected_project, str) else ""
        if not expected:
            # Engram v3 400s it ("expected_project must be a valid non-empty project name").
            self._record_error(
                f"PATCH /observations/{observation_id}: refused locally, blank expected_project"
            )
            return None
        return _as_dict(
            self._request(
                "PATCH",
                f"/observations/{observation_id}",
                params={"expected_project": expected},
                body=body,
            )
        )

    def get_observation(
        self,
        observation_id: int,
        *,
        include_history: bool = False,
        history_cursor: int | None = None,
    ) -> dict[str, Any] | None:
        """``GET /observations/{id}`` → the observation, or ``None`` when missing.

        With ``include_history`` (Engram 3.6) the row also carries ``history`` (at most
        50 prior snapshots), ``version_count`` and, when older pages remain,
        ``history_truncated`` / ``history_from_version`` / ``history_cursor``.
        ``history_cursor`` is only sent with the flag, the only case the server reads
        it; without the flag the request carries no query at all. An older server
        ignores both and answers the plain row.
        """
        params: dict[str, Any] = {}
        if include_history:
            params["include_history"] = True
            if history_cursor is not None:
                params["history_cursor"] = history_cursor
        return _as_dict(self._request("GET", f"/observations/{observation_id}", params=params))

    def recent_observations(
        self, project: str = "", limit: int = 20, scope: str = ""
    ) -> list[dict[str, Any]]:
        """``GET /observations/recent`` → newest observations first; ``[]`` on failure."""
        return _as_list(
            self._request(
                "GET",
                "/observations/recent",
                params={"project": project, "limit": limit, "scope": scope},
            )
        )

    def passive_capture(
        self, session_id: str, content: str, project: str = "", source: str = ""
    ) -> dict[str, Any] | None:
        """``POST /observations/passive`` → ``{extracted, saved, duplicates}`` (HTTP 200)."""
        body: dict[str, Any] = {"session_id": session_id, "content": content}
        for key, value in (("project", project), ("source", source)):
            if value.strip():
                body[key] = value
        return _as_dict(self._request("POST", "/observations/passive", body=body))

    # -- Search and context --------------------------------------------------

    def search(
        self,
        query: str,
        project: str = "",
        limit: int = 10,
        match_mode: str = "any",
        type_: str = "",
        scope: str = "",
        all_projects: bool = False,
    ) -> list[dict[str, Any]]:
        """``GET /search`` → ranked hits; always a list, ``[]`` on failure."""
        if not query.strip():
            # The server 400s on a blank `q`, so there is nothing to ask it.
            self._record_error("GET /search: refused locally, blank query")
            return []

        params: dict[str, Any] = {"q": query, "limit": limit, "type": type_, "scope": scope}

        # `project` and `all_projects` together is a 400, so they are mutually
        # exclusive here, with all_projects winning because it is the explicit ask.
        if all_projects:
            params["all_projects"] = True
        elif project.strip():
            params["project"] = project

        # An unrecognised match_mode is a 400 rather than a fallback, so drop it
        # and let the server apply its own default.
        normalized_mode = match_mode.strip().lower() if isinstance(match_mode, str) else ""
        if normalized_mode in _MATCH_MODES:
            params["match_mode"] = normalized_mode

        return _as_list(self._request("GET", "/search", params=params))

    def context(self, project: str = "", scope: str = "") -> str:
        """``GET /context`` → the formatted context block; ``""`` on failure."""
        return _as_context(
            self._request("GET", "/context", params={"project": project, "scope": scope})
        )

    def compaction_context(self, session_id: str) -> str:
        """``GET /context/compaction`` → the session's compaction block; ``""`` on failure."""
        if not session_id.strip():
            # The server requires session_id.
            self._record_error("GET /context/compaction: refused locally, blank session_id")
            return ""
        return _as_context(
            self._request("GET", "/context/compaction", params={"session_id": session_id})
        )

    # -- Prompts, timeline, stats -------------------------------------------

    def add_prompt(self, session_id: str, content: str, project: str = "") -> dict[str, Any] | None:
        """``POST /prompts`` → ``{"id", "status"}``."""
        missing = [
            name
            for name, value in (("session_id", session_id), ("content", content))
            if not value.strip()
        ]
        if missing:
            # The server requires both.
            self._record_error(
                f"POST /prompts: refused locally, blank required field(s): {', '.join(missing)}"
            )
            return None
        body: dict[str, Any] = {"session_id": session_id, "content": content}
        if project.strip():
            body["project"] = project
        return _as_dict(self._request("POST", "/prompts", body=body))

    def timeline(
        self, observation_id: int, before: int = 5, after: int = 5, project: str = ""
    ) -> dict[str, Any] | None:
        """``GET /timeline`` → the focus observation with its neighbours."""
        return _as_dict(
            self._request(
                "GET",
                "/timeline",
                params={
                    "observation_id": observation_id,
                    "before": before,
                    "after": after,
                    "project": project,
                },
            )
        )

    # -- Conflicts and pins --------------------------------------------------

    def judge_conflict(
        self,
        judgment_id: str,
        relation: str,
        *,
        reason: str = "",
        evidence: str = "",
        confidence: float | None = None,
        session_id: str = "",
    ) -> dict[str, Any] | None:
        """``POST /conflicts/judge`` → ``{"relation": {...}}``.

        ``judgment_id`` and ``relation`` are required by the server; blank optional
        strings are omitted (the server treats ``""`` as "not given" anyway).
        """
        missing = [
            name
            for name, value in (("judgment_id", judgment_id), ("relation", relation))
            if not value.strip()
        ]
        if missing:
            self._record_error(
                "POST /conflicts/judge: refused locally, blank required field(s): "
                + ", ".join(missing)
            )
            return None
        body: dict[str, Any] = {"judgment_id": judgment_id, "relation": relation}
        for key, value in (("reason", reason), ("evidence", evidence), ("session_id", session_id)):
            if value.strip():
                body[key] = value
        if confidence is not None:
            body["confidence"] = confidence
        return _as_dict(self._request("POST", "/conflicts/judge", body=body))

    def judge_many(
        self, judgments: list[dict[str, Any]], session_id: str = ""
    ) -> dict[str, Any] | None:
        """``POST /conflicts/judge-many`` → ``{"outcomes": [...]}``, one per judgment.

        Engram 3.6. The batch is 1..32 items; each item is applied (or refused) on its
        own, so a 200 can carry per-item errors. ``session_id`` replaces every item's
        provenance server-side; a blank one is omitted.
        """
        if not isinstance(judgments, list) or not 1 <= len(judgments) <= JUDGE_MANY_MAX:
            self._record_error(
                f"POST /conflicts/judge-many: refused locally, batch must hold "
                f"1..{JUDGE_MANY_MAX} judgments"
            )
            return None
        body: dict[str, Any] = {"judgments": judgments}
        if session_id.strip():
            body["session_id"] = session_id
        return _as_dict(self._request("POST", "/conflicts/judge-many", body=body))

    def consolidate(
        self,
        ids_to_delete: list[int],
        new_title: str,
        new_content: str,
        expected_project: str,
        *,
        type_: str = "",
        topic_key: str = "",
        confirm: bool = False,
        fingerprint: str = "",
    ) -> dict[str, Any] | None:
        """``POST /observations/consolidate`` → a preview (200) or the applied plan (201).

        Engram 3.6 (agentops.ParseConsolidate). Without ``confirm`` it writes nothing
        and answers ``{"status": "preview", "plan"}``; with ``confirm`` and the plan's
        ``fingerprint`` it soft-deletes the sources into one survivor and answers
        ``{"status": "consolidated", "plan", "survivor", "soft_deleted_ids"}``. The
        server refuses any key outside its list, so only the given ones are sent.
        """
        if confirm and not fingerprint.strip():
            # The server 400s it ("fingerprint is required when confirm is true").
            self._record_error(
                "POST /observations/consolidate: refused locally, confirm without fingerprint"
            )
            return None
        body: dict[str, Any] = {
            "ids_to_delete": list(ids_to_delete),
            "new_title": new_title,
            "new_content": new_content,
            "expected_project": expected_project,
        }
        for key, value in (("type", type_), ("topic_key", topic_key)):
            if value.strip():
                body[key] = value
        if confirm:
            body["confirm"] = True
        if fingerprint.strip():
            body["fingerprint"] = fingerprint
        return _as_dict(self._request("POST", "/observations/consolidate", body=body))

    def pin_observation(self, observation_id: int) -> dict[str, Any] | None:
        """``PUT /observations/{id}/pin`` → ``{"id", "pinned": true}``."""
        return _as_dict(self._request("PUT", f"/observations/{observation_id}/pin"))

    def unpin_observation(self, observation_id: int) -> dict[str, Any] | None:
        """``DELETE /observations/{id}/pin`` → ``{"id", "pinned": false}``."""
        return _as_dict(self._request("DELETE", f"/observations/{observation_id}/pin"))

    def stats(self, project: str = "", all_projects: bool = False) -> dict[str, Any] | None:
        """``GET /stats`` → store counters for one project or the whole server."""
        params: dict[str, Any] = {}
        if all_projects:
            params["all_projects"] = True
        elif project.strip():
            params["project"] = project
        return _as_dict(self._request("GET", "/stats", params=params))


# Hosts that name this machine. Requests to them never need a proxy, and must not
# use one: the bodies carry memory content (prompts, tool results, observations).
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _build_opener(base_url: str) -> urllib.request.OpenerDirector:
    """Return the opener for ``base_url``: proxy-free for loopback, default otherwise.

    urllib's default opener honours ``HTTP_PROXY`` / ``HTTPS_PROXY`` for every
    host, including ``127.0.0.1`` unless ``NO_PROXY`` happens to list it. A local
    Engram is the common case, and routing it through a corporate or debugging
    proxy would hand that proxy every memory write. An explicit empty
    ``ProxyHandler`` replaces the environment-derived one. A remote Engram keeps
    the environment's proxies, since reaching it may genuinely require them.
    """
    try:
        host = (urllib.parse.urlsplit(base_url).hostname or "").lower()
    except ValueError:
        host = ""
    if host in _LOOPBACK_HOSTS:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _is_connection_refused(reason: Any) -> bool:
    """Whether a urllib failure reason is a TCP connection refusal."""
    if isinstance(reason, ConnectionRefusedError):
        return True
    return isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED


def _query_value(value: Any) -> str:
    """Render a query param value, keeping ``0`` and ``False`` as real values."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _http_error_payload(exc: urllib.error.HTTPError) -> dict[str, Any] | None:
    """The HTTPError body as a JSON object, or ``None`` when it is not one."""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _http_error_detail(exc: urllib.error.HTTPError, payload: dict[str, Any] | None) -> str:
    """The server's ``error`` field from the decoded body, else the HTTP reason."""
    if isinstance(payload, dict):
        message = payload.get("error")
        if isinstance(message, str) and message:
            return f" {message}"
    return f" {exc.reason}" if exc.reason else ""


def _as_dict(payload: Any) -> dict[str, Any] | None:
    """Return ``payload`` when it is a dict, else ``None``."""
    return payload if isinstance(payload, dict) else None


def _as_list(payload: Any) -> list[dict[str, Any]]:
    """Return ``payload`` when it is a list, else ``[]`` — these methods always list."""
    return payload if isinstance(payload, list) else []


def _as_context(payload: Any) -> str:
    """Return the ``"context"`` string from a context response, else ``""``."""
    if isinstance(payload, dict):
        context = payload.get("context")
        if isinstance(context, str):
            return context
    return ""
