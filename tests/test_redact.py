"""Behaviour of ``hermes_recall.redact``: private-block redaction and length capping.

Pure functions ported from gentle-engram, so these tests pin parity with upstream as
much as they pin behaviour.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from hermes_recall.redact import (
    PRIVATE_TAG_RE,
    REDACTED,
    TRUNCATION_MARKER,
    strip_private,
    truncate,
)

# -- 1. strip_private ---------------------------------------------------------


def test_strip_private_replaces_a_block() -> None:
    assert strip_private("before <private>secret</private> after") == (f"before {REDACTED} after")


def test_strip_private_replaces_every_block() -> None:
    result = strip_private("a <private>one</private> b <private>two</private> c")

    assert result == f"a {REDACTED} b {REDACTED} c"
    assert result.count(REDACTED) == 2


def test_strip_private_is_not_greedy_across_two_blocks() -> None:
    # A greedy pattern would match from the first opener to the LAST closer and eat
    # "keep me" along with both secrets.
    result = strip_private("<private>one</private>keep me<private>two</private>")

    assert result == f"{REDACTED}keep me{REDACTED}"
    assert "keep me" in result


@pytest.mark.parametrize(
    "raw",
    [
        "<PRIVATE>secret</PRIVATE>",
        "<Private>secret</Private>",
        "<private>secret</PRIVATE>",
    ],
)
def test_strip_private_is_case_insensitive(raw: str) -> None:
    assert strip_private(raw) == REDACTED


def test_strip_private_handles_multiline_content() -> None:
    raw = "head\n<private>line one\nline two\n\nline three</private>\ntail"

    result = strip_private(raw)

    assert result == f"head\n{REDACTED}\ntail"
    assert "line two" not in result


def test_strip_private_leaves_an_unclosed_tag_alone() -> None:
    # Deliberate, and matching upstream: the convention is a paired tag, and treating
    # a lone opener as "redact to end of string" would eat content the user kept.
    raw = "visible <private>never closed"

    assert strip_private(raw) == raw


def test_strip_private_leaves_a_lone_closing_tag_alone() -> None:
    assert strip_private("visible </private> more") == "visible </private> more"


def test_strip_private_strips_surrounding_whitespace() -> None:
    # Upstream's stripPrivateTags is redactPrivateTags(str).trim().
    assert strip_private("   <private>x</private>   ") == REDACTED
    assert strip_private("  \n padded \t ") == "padded"


def test_strip_private_can_empty_a_string_entirely() -> None:
    assert strip_private("<private>everything</private>") == REDACTED
    assert strip_private("   ") == ""


@pytest.mark.parametrize("bad", [None, 7, 3.5, [], {}, object(), True])
def test_strip_private_never_raises_on_a_non_string(bad: Any) -> None:
    assert strip_private(bad) == ""


def test_private_tag_pattern_shape() -> None:
    # Upstream: /<private>[\s\S]*?<\/private>/gi — non-greedy, case-insensitive.
    assert PRIVATE_TAG_RE.search("<private>x</private>") is not None
    assert PRIVATE_TAG_RE.flags & re.IGNORECASE


# -- 2. truncate --------------------------------------------------------------


def test_truncate_leaves_a_short_value_untouched() -> None:
    assert truncate("short", 100) == "short"
    assert TRUNCATION_MARKER not in truncate("short", 100)


def test_truncate_leaves_a_value_exactly_at_the_limit_untouched() -> None:
    # Upstream cuts on `length > max`, so equality is not a cut.
    value = "x" * 10
    assert truncate(value, 10) == value


def test_truncate_cuts_to_the_limit_plus_the_marker() -> None:
    result = truncate("y" * 50, 10)

    assert result == "y" * 10 + "..."
    assert len(result) == 10 + len(TRUNCATION_MARKER)


def test_truncate_uses_three_ascii_dots_not_an_ellipsis() -> None:
    # Upstream's prompt-content truncate uses "..."; the single-character "…" belongs
    # to the separate UI helper in memory-tool-chrome.js. They are not interchangeable.
    result = truncate("z" * 20, 5)

    assert result.endswith("...")
    assert "\u2026" not in result


def test_truncate_with_a_zero_limit() -> None:
    assert truncate("anything", 0) == "..."
    assert truncate("", 0) == ""


def test_truncate_clamps_a_negative_limit_to_zero() -> None:
    # A negative slice index in JS keeps the TAIL of the string, which is never what a
    # cap means, so it is refused rather than mirrored.
    assert truncate("abcdef", -3) == "..."


@pytest.mark.parametrize("bad_limit", [None, "10", 3.5, [], True])
def test_truncate_treats_an_unusable_limit_as_zero(bad_limit: Any) -> None:
    assert truncate("abcdef", bad_limit) == "..."


@pytest.mark.parametrize("bad", [None, 7, 3.5, [], {}, object()])
def test_truncate_never_raises_on_a_non_string(bad: Any) -> None:
    assert truncate(bad, 10) == ""


# -- 3. composition -----------------------------------------------------------


def test_redact_then_truncate_cannot_leak_a_straddling_block() -> None:
    # The order the provider uses. A <private> block that straddles the cut is
    # removed before the cut exists, so the boundary can never strand an opening tag
    # and leave the secret visible.
    raw = "a" * 40 + "<private>" + "s" * 40 + "</private>" + "b" * 40

    result = truncate(strip_private(raw), 50)

    assert "s" * 10 not in result
    assert REDACTED in result
    assert len(result) == 50 + len(TRUNCATION_MARKER)


def test_truncate_then_redact_would_leak_the_same_block() -> None:
    # Proof the order matters, and why the provider deviates from upstream here: cut
    # first and the closing tag is gone, so nothing matches and the secret survives.
    raw = "a" * 40 + "<private>" + "s" * 40 + "</private>" + "b" * 40

    leaked = strip_private(truncate(raw, 60))

    assert "s" * 10 in leaked
    assert REDACTED not in leaked
