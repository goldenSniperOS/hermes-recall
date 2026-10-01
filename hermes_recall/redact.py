"""Private-block redaction and length capping for content on its way to Engram.

Ported from gentle-engram's ``private-redaction.js`` (``redactPrivateTags``) and the
``truncate`` / ``stripPrivateTags`` helpers in its ``index.ts``, so a memory written
from Hermes is shaped like one written from Pi.

Stdlib only, no Hermes import. Both functions are total and exception-free: they sit
on the capture path, and a malformed value must cost a capture, never a traceback.

This is a convenience convention, not a secret scanner — upstream says so in its own
docstring, and the same caveat applies here.
"""

from __future__ import annotations

import re
from typing import Any

# Upstream: ``/<private>[\s\S]*?<\/private>/gi``. Non-greedy, so two blocks in one
# string are replaced independently instead of merging into a single match that
# swallows everything between them. An **unclosed** ``<private>`` is deliberately NOT
# redacted — matching upstream, the convention is a paired tag, and treating a lone
# opener as "redact to end of string" would silently eat content the user meant to
# keep. Closing the tag is the user's half of the bargain.
PRIVATE_TAG_RE = re.compile(r"<private>[\s\S]*?</private>", re.IGNORECASE)

REDACTED = "[REDACTED]"

# Upstream's ``truncate`` marks a cut with three ASCII dots. Note this is a different
# function from the UI helper ``truncateText`` in ``memory-tool-chrome.js``, which
# uses the single-character ellipsis "…" — prompt/observation content and status
# chrome are not the same artifact, so the two markers are not interchangeable.
TRUNCATION_MARKER = "..."


def strip_private(value: str) -> str:
    """Replace every complete ``<private>…</private>`` block with ``[REDACTED]``, then strip.

    Mirrors upstream's ``stripPrivateTags``, which is
    ``redactPrivateTags(str).trim()``.

    Returns ``""`` for a non-string rather than raising, so a caller handing in
    whatever the host gave it still gets a usable answer.
    """
    if not isinstance(value, str):
        return ""
    try:
        return PRIVATE_TAG_RE.sub(REDACTED, value).strip()
    except Exception:
        return ""


def truncate(value: str, limit: int) -> str:
    """Cap ``value`` at ``limit`` characters, appending ``"..."`` when it had to cut.

    Mirrors upstream's ``truncate`` in ``index.ts``:
    ``str.length > max ? `${str.slice(0, max)}...` : str``. A value that fits is
    returned byte-identical, with no marker.

    A negative ``limit`` is clamped to ``0`` — upstream would hand it to
    ``String.slice``, where a negative index counts from the end and silently keeps
    the *tail* of the string. That is never what a cap means, so it is refused here.
    """
    if not isinstance(value, str):
        return ""
    try:
        ceiling = _as_limit(limit)
        if len(value) <= ceiling:
            return value
        return f"{value[:ceiling]}{TRUNCATION_MARKER}"
    except Exception:
        return ""


def _as_limit(limit: Any) -> int:
    """Coerce ``limit`` to a non-negative int, treating anything unusable as ``0``."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        return 0
    return max(0, limit)
