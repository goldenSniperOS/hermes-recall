"""The compact ``mem_*`` tool schemas and their handlers, replacing the Engram MCP surface.

Ten tools, each a thin mapping onto one Engram HTTP route. Descriptions are kept
short on purpose: every schema is sent on every API call, and the whole point of
this provider over the Engram MCP server is that the model pays for ten compact
schemas instead of ~19 verbose ones.

Contract of :func:`handle_tool_call`: it **always** returns a JSON string and never
raises. Unknown tool, bad arguments, writes refused, Engram unreachable — each one
is a JSON object with an ``error`` key the model can read and act on.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .client import EngramClient
from .format import (
    observation_status,
    pinned_status,
    render_observation_detail,
    render_status_line,
    saved_status,
    unpinned_status,
    updated_status,
)
from .redact import strip_private, truncate
from .server import read_server_state
from .state import read_state, write_state

logger = logging.getLogger(__name__)

# -- Caps ---------------------------------------------------------------------

SEARCH_LIMIT_DEFAULT = 10
SEARCH_LIMIT_MAX = 20
# Per-hit content in a mem_search result. Search is for finding; the full body is
# one mem_get_observation away, so a hit carries a snippet, not a document.
SEARCH_SNIPPET_MAX_CHARS = 300
# Full observation body in a mem_get_observation result.
OBSERVATION_CONTENT_MAX_CHARS = 12000
CONTEXT_MAX_CHARS = 6000
TIMELINE_WINDOW_MAX = 20
QUERY_MAX_CHARS = 500

JUDGE_RELATIONS = (
    "related",
    "compatible",
    "scoped",
    "conflicts_with",
    "supersedes",
    "not_conflict",
)

# Keys of a save response that indicate Engram wants a conflict verdict. Engram's
# HTTP ``POST /observations`` currently answers ``{"id", "status"}`` only, but the
# MCP surface returns judgment candidates, and a server that starts returning them
# here must not have them silently dropped: every unknown response key is passed
# through, and these ones also earn an explicit next-step hint.
JUDGMENT_KEYS = ("judgment_required", "judgment_id", "candidates", "conflicts")

# Engram's error text for a write naming a session it has no row for.
SESSION_NOT_FOUND = "session not found"

# -- Schemas ------------------------------------------------------------------


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str]):
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _s(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _i(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


_ID = _i("Observation id")
_PROJECT = _s("Project (default: current)")

_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    _schema(
        "mem_search",
        "Search Engram memory. Returns id, type, title and a snippet per hit.",
        {
            "query": _s("Search text"),
            "type": _s("Filter by type"),
            "scope": _s("project | personal | global"),
            "limit": _i(f"Max hits (default {SEARCH_LIMIT_DEFAULT}, max {SEARCH_LIMIT_MAX})"),
            "match_mode": {"type": "string", "enum": ["any", "all"]},
            "project": _PROJECT,
            "all_projects": {"type": "boolean", "description": "Search every project"},
        },
        ["query"],
    ),
    _schema(
        "mem_save",
        "Save an observation to Engram. Content: What/Why/Where/Learned.",
        {
            "title": _s("Verb + what, short"),
            "content": _s("Observation body"),
            "type": _s("bugfix|decision|architecture|discovery|pattern|config|preference"),
            "scope": _s("project (default) | personal | global"),
            "topic_key": _s("Stable key for evolving topics (upsert)"),
            "project": _PROJECT,
        },
        ["title", "content"],
    ),
    _schema(
        "mem_update",
        "Update fields of an existing observation.",
        {
            "id": _ID,
            # Required since Engram v3 (e8c0ce3): the caller asserts the owner, as
            # gentle-engram 0.2.0 does. Never defaulted to the detected project.
            "expected_project": _s(
                "Project that owns the observation; must match or the update is refused"
            ),
            "title": _s("New title"),
            "content": _s("New content"),
            "type": _s("New type"),
            "scope": _s("New scope"),
            "topic_key": _s("New topic key"),
        },
        ["id", "expected_project"],
    ),
    _schema("mem_get_observation", "Get one observation in full.", {"id": _ID}, ["id"]),
    _schema(
        "mem_context",
        "Recent project context from Engram (sessions and observations).",
        {"project": _PROJECT, "scope": _s("project | personal | global")},
        [],
    ),
    _schema(
        "mem_session_summary",
        "Save the session summary: Goal, Discoveries, Accomplished, Next Steps, Files.",
        {"content": _s("Full session summary"), "project": _PROJECT},
        ["content"],
    ),
    _schema(
        "mem_timeline",
        "Observations around one id, in time order.",
        {
            "observation_id": _ID,
            "before": _i("Count before (default 5)"),
            "after": _i("Count after (default 5)"),
            "project": _PROJECT,
        },
        ["observation_id"],
    ),
    _schema(
        "mem_judge",
        "Resolve a memory conflict candidate returned by mem_save.",
        {
            "judgment_id": _s("judgment_id from the candidate"),
            "relation": {"type": "string", "enum": list(JUDGE_RELATIONS)},
            "reason": _s("Why"),
            "evidence": _s("Supporting evidence"),
            "confidence": {"type": "number", "description": "0.0..1.0"},
        },
        ["judgment_id", "relation"],
    ),
    _schema("mem_pin", "Pin an observation so recall favors it.", {"id": _ID}, ["id"]),
    _schema("mem_unpin", "Unpin an observation.", {"id": _ID}, ["id"]),
)

TOOL_NAMES: frozenset[str] = frozenset(schema["name"] for schema in _TOOL_SCHEMAS)
WRITE_TOOLS: frozenset[str] = frozenset(
    {"mem_save", "mem_update", "mem_session_summary", "mem_judge", "mem_pin", "mem_unpin"}
)


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return a fresh copy of the ten schemas. Pure, cheap, no I/O.

    A copy, because the host may normalise or annotate what it is handed, and the
    module constant must stay byte-stable across calls.
    """
    return copy.deepcopy(list(_TOOL_SCHEMAS))


# -- Environment ----------------------------------------------------------------


@dataclass
class ToolEnv:
    """What a handler needs from the provider, passed in so tools never import it."""

    client: EngramClient
    session_id: str
    hermes_home: str
    writes_enabled: bool
    agent_context: str
    resolve_project: Callable[[], str]
    register_session: Callable[[str], bool]
    # The Engram session id writes go to, resolved at call time: ``session_id`` is
    # the Hermes runtime id, and once Engram v3 resumed it as ``<id>:resume:<N>``
    # the registration's acknowledged id is the one to write under. ``None`` (no
    # resolver) means the runtime id itself.
    effective_session_id: Callable[[], str] | None = None

    def write_session_id(self) -> str:
        """The session id to attribute a write to right now."""
        if self.effective_session_id is None:
            return self.session_id
        return self.effective_session_id() or self.session_id


class ToolError(Exception):
    """A user-facing tool failure; its message becomes the JSON ``error``."""


# -- Dispatch -------------------------------------------------------------------


def handle_tool_call(env: ToolEnv, tool_name: str, args: Any) -> str:
    """Run one tool and return its JSON result. Never raises."""
    try:
        if tool_name not in TOOL_NAMES:
            return error_result(f"unknown tool: {tool_name}")
        if args is None:
            args = {}
        if isinstance(args, str):
            # Some hosts hand over the raw JSON arguments string.
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return error_result("arguments must be a JSON object")
        if not isinstance(args, dict):
            return error_result("arguments must be a JSON object")
        if tool_name in WRITE_TOOLS and not env.writes_enabled:
            return error_result(
                "Engram writes are disabled for subagents and other non-primary "
                f"contexts (agent_context={env.agent_context}); read tools still work."
            )
        handler = _HANDLERS[tool_name]
        return _dump(handler(env, args))
    except ToolError as exc:
        return error_result(str(exc))
    except Exception as exc:  # noqa: BLE001 - the contract is "never raise"
        logger.debug("recall tool %s failed", tool_name, exc_info=True)
        return error_result(f"{tool_name} failed: {type(exc).__name__}")


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def error_result(message: str) -> str:
    return _dump({"error": message})


def _engram_error(env: ToolEnv, action: str) -> ToolError:
    reason = env.client.last_error_text() or "no response"
    return ToolError(f"Engram {action} failed: {reason}. Is `engram serve` running?")


# -- Argument parsing -----------------------------------------------------------


def _str_arg(args: dict[str, Any], key: str, *, required: bool = False) -> str:
    value = args.get(key)
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ToolError(f"'{key}' must be a string")
    value = value.strip()
    if required and not value:
        raise ToolError(f"'{key}' is required")
    return value


def _id_arg(args: dict[str, Any], key: str) -> int:
    value = args.get(key)
    if isinstance(value, str) and value.strip().lstrip("#").isdigit():
        value = int(value.strip().lstrip("#"))
    elif isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ToolError(f"'{key}' must be a positive integer observation id")
    return value


def _int_arg(args: dict[str, Any], key: str, *, default: int, minimum: int, maximum: int) -> int:
    value = args.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    elif isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"'{key}' must be an integer")
    return max(minimum, min(maximum, value))


def _bool_arg(args: dict[str, Any], key: str) -> bool:
    value = args.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _project(env: ToolEnv, args: dict[str, Any]) -> str:
    return _str_arg(args, "project") or env.resolve_project() or ""


def _remember(env: ToolEnv, observation_id: Any, action: str, project: str) -> None:
    """Persist the last memory id. A failure is swallowed: state is a convenience.

    Primary context only. ``state.json`` describes what the user's own agent did;
    a subagent or cron run reading an observation must not move ``last_memory_id``
    under the plugins that read it.
    """
    try:
        if not env.writes_enabled:
            return
        if isinstance(observation_id, bool) or not isinstance(observation_id, int):
            return
        write_state(
            env.hermes_home,
            last_memory_id=observation_id,
            last_action=action,
            project=project or "",
        )
    except Exception:
        logger.debug("recall: state write failed (non-fatal)", exc_info=True)


def _observation_id(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("id")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: hit.get(key)
        for key in ("id", "type", "title", "project", "scope", "topic_key", "created_at")
        if hit.get(key) not in (None, "")
    }
    content = hit.get("content")
    if isinstance(content, str) and content:
        compact["snippet"] = truncate(" ".join(content.split()), SEARCH_SNIPPET_MAX_CHARS)
    return compact


# -- Handlers -------------------------------------------------------------------


def _mem_search(env: ToolEnv, args: dict[str, Any]) -> dict[str, Any]:
    query = truncate(_str_arg(args, "query", required=True), QUERY_MAX_CHARS)
    all_projects = _bool_arg(args, "all_projects")
    project = "" if all_projects else _project(env, args)
    match_mode = _str_arg(args, "match_mode").lower() or "any"
    if match_mode not in {"any", "all"}:
        raise ToolError("'match_mode' must be 'any' or 'all'")
    hits = env.client.search(
        query=query,
        project=project,
        limit=_int_arg(
            args, "limit", default=SEARCH_LIMIT_DEFAULT, minimum=1, maximum=SEARCH_LIMIT_MAX
        ),
        match_mode=match_mode,
        type_=_str_arg(args, "type"),
        scope=_str_arg(args, "scope"),
        all_projects=all_projects,
    )
    if not hits and env.client.last_error_text():
        raise _engram_error(env, "search")
    results = [_compact_hit(hit) for hit in hits if isinstance(hit, dict)]
    return {
        "project": project or ("*" if all_projects else ""),
        "count": len(results),
        "results": results,
    }


def _save(
    env: ToolEnv,
    *,
    title: str,
    content: str,
    type_: str,
    scope: str,
    topic_key: str,
    project: str,
) -> dict[str, Any]:
    """POST /observations, registering the session once and retrying on a 404."""
    if not env.session_id:
        raise ToolError("no active Hermes session to save under")

    def post() -> dict[str, Any] | None:
        # Resolved per attempt: the retry's registration may have selected a
        # continuation of the session.
        return env.client.add_observation(
            env.write_session_id(),
            title,
            content,
            type_=type_,
            project=project,
            scope=scope,
            topic_key=topic_key,
        )

    saved = with_session_retry(env.client, post, lambda: env.register_session(project))
    if saved is None:
        raise _engram_error(env, "save")
    return saved


def with_session_retry(
    client: EngramClient, write: Callable[[], Any], register: Callable[[], bool]
) -> Any:
    """Run ``write``; on Engram's ``session not found``, ``register`` once and retry once.

    Shared by the tools and every background write in the provider. Engram answers
    404 ``session not found`` whenever a write names a project and the session row
    does not exist yet (``validateSessionProject``, server.go) — which is what
    happens when the registration from ``initialize`` never landed. Registration is
    idempotent server-side, so the recovery is a single register-then-retry. One
    retry, never a loop: a second 404 means something else is wrong.
    """
    result = write()
    if result is None and SESSION_NOT_FOUND in client.last_error_text():
        if register():
            result = write()
    return result


def _save_result(saved: dict[str, Any], project: str) -> dict[str, Any]:
    observation_id = _observation_id(saved)
    result: dict[str, Any] = {}
    for key, value in saved.items():
        if key == "status":
            result["engram_status"] = value
        else:
            result[key] = value
    result["id"] = observation_id
    result["project"] = project
    result["status"] = saved_status(observation_id) if observation_id else "🧠 saved"
    if any(saved.get(key) for key in JUDGMENT_KEYS):
        result["next_step"] = (
            "Engram flagged possible conflicts: call mem_judge with each candidate's "
            "judgment_id and a relation."
        )
    return result


def _mem_save(env: ToolEnv, args: dict[str, Any]) -> dict[str, Any]:
    title = strip_private(_str_arg(args, "title", required=True))
    content = strip_private(_str_arg(args, "content", required=True))
    if not title or not content:
        raise ToolError("'title' and 'content' must not be empty after redaction")
    project = _project(env, args)
    saved = _save(
        env,
        title=title,
        content=content,
        type_=_str_arg(args, "type") or "manual",
        scope=_str_arg(args, "scope") or "project",
        topic_key=_str_arg(args, "topic_key"),
        project=project,
    )
    result = _save_result(saved, project)
    _remember(env, result["id"], "saved", project)
    return result


def _mem_session_summary(env: ToolEnv, args: dict[str, Any]) -> dict[str, Any]:
    content = strip_private(_str_arg(args, "content", required=True))
    if not content:
        raise ToolError("'content' must not be empty after redaction")
    project = _project(env, args)
    saved = _save(
        env,
        title="Session summary",
        content=content,
        type_="session_summary",
        scope="project",
        topic_key="",
        project=project,
    )
    result = _save_result(saved, project)
    _remember(env, result["id"], "session_summary", project)
    return result


def _mem_update(env: ToolEnv, args: dict[str, Any]) -> dict[str, Any]:
    observation_id = _id_arg(args, "id")
    expected_project = _str_arg(args, "expected_project")
    if not expected_project:
        raise ToolError(
            "'expected_project' is required: the project that owns the observation "
            f"(mem_get_observation #{observation_id} shows it)"
        )
    fields: dict[str, Any] = {}
    for key in ("title", "content", "type", "scope", "topic_key"):
        if args.get(key) is None:
            continue
        value = _str_arg(args, key)
        fields[key] = strip_private(value) if key in ("title", "content") else value
    if not fields:
        raise ToolError("give at least one of: title, content, type, scope, topic_key")
    updated = env.client.update_observation(
        observation_id, expected_project=expected_project, **fields
    )
    if updated is None:
        if env.client.last_error_status is not None:
            # A server answered (409 owner mismatch, 400 invalid patch, 404, ...):
            # asking the model whether `engram serve` is running would send it the
            # wrong way.
            reason = env.client.last_error_text()
            raise ToolError(f"Engram refused the update of #{observation_id}: {reason}")
        raise _engram_error(env, "update")
    project = updated.get("project") if isinstance(updated.get("project"), str) else ""
    _remember(env, observation_id, "updated", project or env.resolve_project())
    return {
        "id": observation_id,
        "status": updated_status(observation_id),
        "observation": _compact_hit(updated),
    }


def _mem_get_observation(env: ToolEnv, args: dict[str, Any]) -> dict[str, Any]:
    observation_id = _id_arg(args, "id")
    observation = env.client.get_observation(observation_id)
    if observation is None:
        raise _engram_error(env, f"get #{observation_id}")
    body = dict(observation)
    content = body.get("content")
    if isinstance(content, str) and len(content) > OBSERVATION_CONTENT_MAX_CHARS:
        body["content"] = truncate(content, OBSERVATION_CONTENT_MAX_CHARS)
        body["content_truncated"] = True
    project = body.get("project") if isinstance(body.get("project"), str) else ""
    _remember(env, observation_id, "observation", project)
    return {"id": observation_id, "status": observation_status(observation_id), "observation": body}


def _mem_context(env: ToolEnv, args: dict[str, Any]) -> dict[str, Any]:
    project = _project(env, args)
    context = env.client.context(project=project, scope=_str_arg(args, "scope"))
    if not context and env.client.last_error_text():
        raise _engram_error(env, "context")
    return {"project": project, "context": truncate(context, CONTEXT_MAX_CHARS)}


def _mem_timeline(env: ToolEnv, args: dict[str, Any]) -> dict[str, Any]:
    observation_id = _id_arg(args, "observation_id")
    window = {"default": 5, "minimum": 0, "maximum": TIMELINE_WINDOW_MAX}
    timeline = env.client.timeline(
        observation_id,
        before=_int_arg(args, "before", **window),
        after=_int_arg(args, "after", **window),
        project=_project(env, args),
    )
    if timeline is None:
        raise _engram_error(env, "timeline")
    return timeline


def _mem_judge(env: ToolEnv, args: dict[str, Any]) -> dict[str, Any]:
    judgment_id = _str_arg(args, "judgment_id", required=True)
    relation = _str_arg(args, "relation", required=True)
    if relation not in JUDGE_RELATIONS:
        raise ToolError(f"'relation' must be one of: {', '.join(JUDGE_RELATIONS)}")
    confidence = args.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise ToolError("'confidence' must be a number between 0.0 and 1.0")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ToolError("'confidence' must be between 0.0 and 1.0")
        confidence = float(confidence)
    judged = env.client.judge_conflict(
        judgment_id,
        relation,
        reason=_str_arg(args, "reason"),
        evidence=_str_arg(args, "evidence"),
        confidence=confidence,
        session_id=env.write_session_id(),
    )
    if judged is None:
        raise _engram_error(env, "judge")
    return {**judged, "status": f"🧠 judged {judgment_id}: {relation}"}


def _pin_handler(pinned: bool) -> Callable[[ToolEnv, dict[str, Any]], dict[str, Any]]:
    def handler(env: ToolEnv, args: dict[str, Any]) -> dict[str, Any]:
        observation_id = _id_arg(args, "id")
        call = env.client.pin_observation if pinned else env.client.unpin_observation
        response = call(observation_id)
        if response is None:
            raise _engram_error(env, "pin" if pinned else "unpin")
        _remember(env, observation_id, "pinned" if pinned else "unpinned", env.resolve_project())
        status = pinned_status(observation_id) if pinned else unpinned_status(observation_id)
        return {"id": observation_id, "pinned": pinned, "status": status}

    return handler


_HANDLERS: dict[str, Callable[[ToolEnv, dict[str, Any]], Any]] = {
    "mem_search": _mem_search,
    "mem_save": _mem_save,
    "mem_update": _mem_update,
    "mem_get_observation": _mem_get_observation,
    "mem_context": _mem_context,
    "mem_session_summary": _mem_session_summary,
    "mem_timeline": _mem_timeline,
    "mem_judge": _mem_judge,
    "mem_pin": _pin_handler(True),
    "mem_unpin": _pin_handler(False),
}

# -- Human status (``/recall`` and ``hermes recall status``) ---------------------


def status_report(
    client: EngramClient,
    hermes_home: str,
    *,
    project: str = "",
    recall_count: int | None = None,
    server_status: Any = None,
) -> str:
    """Probe Engram health and render the shared plain-text status block. Never raises.

    ``server_status`` is the in-process auto-start outcome (a ``ServerStatus``); when
    it is ``None`` the outcome recorded in ``<hermes_home>/recall/server.json`` is
    used, which is how ``hermes recall status`` — a separate process — knows whether
    recall started the server.
    """
    try:
        state = read_state(hermes_home)
        lifecycle = server_status if server_status is not None else read_server_state(hermes_home)
        health = client.health()
        server_ok = isinstance(health, dict) and health.get("status") == "ok"
        last_error = "" if server_ok else (client.last_error_text() or "unexpected /health reply")
        return render_status_line(
            project or state.get("project", ""),
            server_ok=server_ok,
            last_error=last_error,
            last_memory_id=state.get("last_memory_id"),
            last_action=state.get("last_action", ""),
            recall_count=recall_count,
            url=client.base_url,
            server_origin=getattr(lifecycle, "status", "") or "",
            server_reason=getattr(lifecycle, "reason", "") or "",
        )
    except Exception:
        return "🧠 Recall — status unavailable"


def observation_report(client: EngramClient, raw_id: str) -> str:
    """Fetch one observation and render the shared detail view. Never raises."""
    try:
        try:
            observation_id = _id_arg({"id": raw_id}, "id")
        except ToolError:
            return f"🧠 Recall — not an observation id: {raw_id.strip()!r}"
        observation = client.get_observation(observation_id)
        return render_observation_detail(observation, error=client.last_error_text())
    except Exception:
        return "🧠 Recall — observation not available"
