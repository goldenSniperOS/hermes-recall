"""Rendering: the injected recall block, the 🧠 recall status line and saved-id notices.

Pure functions only — no I/O, no Hermes import, no logging. Every function here is
total and exception-free: they run on the turn's critical path, and a rendering bug
must degrade to "less text", never to a traceback.
"""

from __future__ import annotations

from typing import Any

# Heading of the injected recall block. Deliberately plain Markdown: Hermes wraps
# whatever a provider returns in its own ``<memory-context>`` envelope with a
# system note (``build_memory_context_block``), and ``sanitize_context`` strips —
# and logs a warning about — any provider that pre-wraps its own tags. So we emit
# plain text and let the host do the wrapping. Keep the wording stable: it ends up
# in the prompt prefix, and churn there costs prompt-cache hits.
RECALL_HEADER = "## Recalled memory (Engram)"

# Per-hit snippet ceiling. Observation content is multi-line Markdown that can run
# to kilobytes; without a cap one hit would eat the whole recall budget.
SNIPPET_MAX_CHARS = 240

# Rendered for any value the caller does not know yet.
UNKNOWN = "—"


def render_hit(hit: dict) -> str:
    """Render one Engram search hit as a single line, or ``""`` when unusable.

    Shape: ``#<id> [<type>] <title>: <snippet>``.

    The snippet comes from ``hit["content"]``. Verified against a live Engram
    2.0.0 server: search hits carry **no** ``snippet`` field. Their real fields are
    ``id`` (int), ``sync_id``, ``session_id``, ``type``, ``title``, ``content``,
    ``project``, ``scope``, ``topic_key``, ``revision_count``, ``duplicate_count``,
    ``last_seen_at``, ``created_at``, ``updated_at`` and ``rank`` (float).

    A hit with no usable ``id`` renders as ``""`` — the id is the whole point of
    the line (the user resumes work by memory id), so a line without one is worse
    than no line. ``type`` falls back to ``memory`` and ``title`` to
    ``(untitled)``; the ``: <snippet>`` suffix is omitted when the content is empty.
    """
    try:
        if not isinstance(hit, dict):
            return ""
        identifier = _hit_id(hit.get("id"))
        if not identifier:
            return ""
        type_ = _text_field(hit, "type", "memory")
        title = _text_field(hit, "title", "(untitled)")
        snippet = _snippet(hit.get("content"))
        line = f"#{identifier} [{type_}] {title}"
        return f"{line}: {snippet}" if snippet else line
    except Exception:
        return ""


def render_recall_block(hits: Any, context: str = "", *, max_chars: int) -> tuple[str, int]:
    """Render the per-turn recall block, returning ``(text, hits_included)``.

    Layout: the header, then the optional ``context`` section (Engram's broader
    project context, fetched once per session), then one line per hit.

    Budget policy — the returned string is **never** longer than ``max_chars``:

    * The header must fit, or nothing is rendered at all.
    * At most **half** the budget is reserved for the header plus the context
      section. A context longer than that is truncated with ``…`` rather than
      allowed to crowd out every hit, because the per-turn hits are the part that
      answers *this* turn.
    * Hits are then added one at a time and the loop stops *before* the budget
      would be exceeded. A hit that does not fit is dropped whole — never
      truncated mid-line, which would leave a half-written memory id.
    * ``("", 0)`` when there is nothing to render, and also when only the header
      would survive: a header-only block is not memory, but Hermes treats any
      non-empty return as "memory was injected" and renders an indicator for it.

    The returned count is the number of **hits** included and excludes the context
    section, because it feeds ``RecallStatus.count`` — "recalled N memories".
    """
    try:
        lines = [line for line in (render_hit(hit) for hit in _as_iterable(hits)) if line]
        context_text = context.strip() if isinstance(context, str) else ""
        if not lines and not context_text:
            return ("", 0)

        budget = int(max_chars) if isinstance(max_chars, int) else 0
        body = RECALL_HEADER
        if budget <= 0 or len(body) > budget:
            return ("", 0)

        context_section = _context_section(context_text, header_len=len(body), budget=budget)
        body += context_section

        count = 0
        for line in lines:
            separator = "\n\n" if count == 0 else "\n"
            candidate = f"{body}{separator}{line}"
            if len(candidate) > budget:
                break
            body = candidate
            count += 1

        if count == 0 and not context_section:
            return ("", 0)
        return (body, count)
    except Exception:
        return ("", 0)


def render_status_line(
    project: str,
    *,
    server_ok: bool,
    last_error: str = "",
    last_memory_id: int | None = None,
    last_action: str = "",
    recall_count: int | None = None,
    url: str = "",
    server_origin: str = "",
    server_reason: str = "",
) -> str:
    """Render the human-facing recall status block (``hermes recall status``, ``/recall``).

    Plain text, deterministic, multi-line, with 🧠 as the only emoji (the glyph
    Hermes already uses for memory indicators). Every value the caller does not
    know renders as ``—`` rather than as a blank or a guess.

    ``url`` is optional and additive: the health verdict is meaningful on its own,
    but when the caller knows which server it probed it belongs on the line.

    ``server_origin`` is the auto-start outcome (``started``, ``adopted``,
    ``foreign``, ``external``, ``disabled``, ``unavailable``; see ``server.py``) and
    ``server_reason`` its human reason. Both are optional: without an origin the line
    reads exactly as it did before auto-start existed.
    """
    try:
        lines = [
            "🧠 Recall — Engram memory provider",
            f"  project:      {_or_unknown(project)}",
            f"  server:       {_server_verdict(url, server_ok, server_origin, server_reason)}",
            f"  last error:   {_or_unknown(last_error)}",
            f"  last memory:  {_memory_label(last_memory_id)}",
            f"  last action:  {_or_unknown(last_action)}",
            f"  last recall:  {_recall_label(recall_count)}",
        ]
        return "\n".join(lines)
    except Exception:
        return "🧠 Recall — status unavailable"


def saved_status(observation_id: Any) -> str:
    """Render the write confirmation for a new observation.

    Mirrors gentle-engram's ``compactResultStatus`` in ``memory-tool-chrome.js``,
    which renders ``✓ saved #<id>``; the glyph is swapped for the 🧠 Hermes already
    uses for memory indicators, so recall and capture read as one feature.
    """
    return f"🧠 saved #{observation_id}"


def updated_status(observation_id: Any) -> str:
    """Render the write confirmation for an updated observation (``✓ updated #<id>`` upstream)."""
    return f"🧠 updated #{observation_id}"


def observation_status(observation_id: Any) -> str:
    """Render the read confirmation for a fetched observation."""
    return f"🧠 observation #{observation_id}"


def pinned_status(observation_id: Any) -> str:
    """Render the confirmation for a pinned observation."""
    return f"🧠 pinned #{observation_id}"


def unpinned_status(observation_id: Any) -> str:
    """Render the confirmation for an unpinned observation."""
    return f"🧠 unpinned #{observation_id}"


# Content ceiling of the ``/recall <id>`` and ``hermes recall show <id>`` detail view.
DETAIL_CONTENT_MAX_CHARS = 800


def render_observation_detail(
    observation: Any, *, max_chars: int = DETAIL_CONTENT_MAX_CHARS, error: str = ""
) -> str:
    """Render one observation for a human: id, title, type, created_at and content.

    Plain text for the ``/recall <id>`` slash command and the CLI. The content keeps
    its line breaks (it is read by a person, not packed into a prompt) and is cut at
    ``max_chars`` with ``…``. A missing observation renders a one-line notice that
    carries ``error`` when the caller has one.
    """
    try:
        if not isinstance(observation, dict) or not _hit_id(observation.get("id")):
            detail = f" ({error.strip()})" if isinstance(error, str) and error.strip() else ""
            return f"🧠 Recall — observation not available{detail}"
        identifier = _hit_id(observation.get("id"))
        content = observation.get("content")
        text = content.strip() if isinstance(content, str) else ""
        limit = max_chars if isinstance(max_chars, int) and max_chars > 0 else 0
        if len(text) > limit:
            text = text[: max(0, limit - 1)].rstrip() + "\u2026"
        lines = [
            f"🧠 observation #{identifier}",
            f"  title:    {_or_unknown(observation.get('title'))}",
            f"  type:     {_or_unknown(observation.get('type'))}",
            f"  project:  {_or_unknown(observation.get('project'))}",
            f"  created:  {_or_unknown(observation.get('created_at'))}",
            "",
            text or UNKNOWN,
        ]
        return "\n".join(lines)
    except Exception:
        return "🧠 Recall — observation not available"


# -- internals ----------------------------------------------------------------


def _as_iterable(hits: Any) -> list[Any]:
    """Return ``hits`` as a list, or ``[]`` for anything that is not a sequence of hits."""
    if isinstance(hits, list):
        return hits
    if isinstance(hits, tuple):
        return list(hits)
    return []


def _hit_id(value: Any) -> str:
    """Render a hit's id, or ``""`` when there is nothing usable.

    Engram returns an ``int``; a numeric string is accepted because a JSON
    producer is free to widen it. Bools are rejected explicitly — ``isinstance(
    True, int)`` holds in Python, and ``#True`` is not a memory id.
    """
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        return text if text.isdigit() else ""
    return ""


def _text_field(hit: dict, key: str, fallback: str) -> str:
    """Return a stripped string field, or ``fallback`` when missing/blank/non-string."""
    value = hit.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _snippet(content: Any) -> str:
    """Collapse whitespace runs in observation content and cap it at ``SNIPPET_MAX_CHARS``.

    Collapsing is not cosmetic: content is multi-line Markdown (``**What**:`` /
    ``**Why**:`` blocks), and keeping its newlines would break the one-line-per-hit
    layout the block depends on.
    """
    if not isinstance(content, str):
        return ""
    collapsed = " ".join(content.split())
    if len(collapsed) <= SNIPPET_MAX_CHARS:
        return collapsed
    return collapsed[: SNIPPET_MAX_CHARS - 1].rstrip() + "\u2026"


def _context_section(context_text: str, *, header_len: int, budget: int) -> str:
    """Render the context section, truncated so header+context stay within half the budget."""
    if not context_text:
        return ""
    allowance = budget // 2
    section = f"\n\n{context_text}"
    if header_len + len(section) <= allowance:
        return section
    # 2 for the "\n\n" separator, 1 for the ellipsis.
    room = allowance - header_len - 3
    if room <= 0:
        return ""
    return f"\n\n{context_text[:room]}\u2026"


def _or_unknown(value: Any) -> str:
    """Return a stripped string, or ``—`` when there is nothing to show."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return UNKNOWN


# How the server was obtained, appended to "reachable" (keys mirror server.py).
_ORIGIN_LABELS = {
    "started": "started by recall",
    "adopted": "adopted",
    "external": "externally managed",
    "disabled": "autostart off",
}


def _server_verdict(url: str, server_ok: bool, origin: str = "", reason: str = "") -> str:
    """Render the server health verdict, prefixed with the URL when one is known.

    Reachability is the live probe; the origin is how recall obtained the server.
    A server that is down after a failed start says why (``not running — <reason>``).
    """
    origin = origin.strip() if isinstance(origin, str) else ""
    reason = reason.strip() if isinstance(reason, str) else ""
    if server_ok:
        if origin == "foreign":
            # A definite identity mismatch: the provider sends it no memory.
            verdict = (
                f"reachable, foreign instance — not used: {reason}"
                if reason
                else "reachable, foreign instance — not used"
            )
        elif origin == "unavailable":
            verdict = f"reachable, not verified — {reason}" if reason else "reachable, not verified"
        elif origin in _ORIGIN_LABELS:
            verdict = f"reachable, {_ORIGIN_LABELS[origin]}"
        else:
            verdict = "reachable"
    elif origin == "unavailable":
        verdict = f"not running — {reason}" if reason else "not running"
    else:
        verdict = "unreachable"
    cleaned = url.strip() if isinstance(url, str) else ""
    return f"{cleaned} ({verdict})" if cleaned else verdict


def _memory_label(last_memory_id: Any) -> str:
    """Render the last memory id as ``#<id>``, or ``—`` when unknown."""
    identifier = _hit_id(last_memory_id)
    return f"#{identifier}" if identifier else UNKNOWN


def _recall_label(recall_count: Any) -> str:
    """Render the last recall count in words, or ``—`` when unknown."""
    if isinstance(recall_count, bool) or not isinstance(recall_count, int):
        return UNKNOWN
    if recall_count == 1:
        return "1 memory"
    return f"{recall_count} memories"
