"""Behaviour of ``hermes_recall.format``: hit lines, the budgeted recall block, status text.

Pure rendering, so nothing here touches a server or a provider.
"""

from __future__ import annotations

from typing import Any

import pytest

from hermes_recall.format import (
    RECALL_HEADER,
    SNIPPET_MAX_CHARS,
    render_hit,
    render_recall_block,
    render_status_line,
    saved_status,
    updated_status,
)

# -- helpers ------------------------------------------------------------------


def hit(**over: Any) -> dict[str, Any]:
    """A search hit shaped like a live Engram 2.0.0 answer (content, never snippet)."""
    base: dict[str, Any] = {
        "id": 761,
        "sync_id": "obs-abc",
        "session_id": "sess-1",
        "type": "decision",
        "title": "Chose Engram over MCP",
        "content": "**What**: switched to the provider route",
        "project": "hermes-recall",
        "scope": "project",
        "topic_key": "",
        "rank": -1.5,
    }
    base.update(over)
    return base


# -- 1. render_hit ------------------------------------------------------------


def test_render_hit_shape() -> None:
    assert render_hit(hit(id=761, type="decision", title="Title", content="snippet")) == (
        "#761 [decision] Title: snippet"
    )


def test_render_hit_collapses_whitespace_runs() -> None:
    # Observation content is multi-line markdown; newlines would wreck the
    # one-line-per-hit layout the block depends on.
    rendered = render_hit(hit(content="**What**: a thing\n\n**Why**:   because\tof\nthis"))
    assert rendered == (
        "#761 [decision] Chose Engram over MCP: **What**: a thing **Why**: because of this"
    )
    assert "\n" not in rendered


def test_render_hit_truncates_a_long_snippet_with_an_ellipsis() -> None:
    rendered = render_hit(hit(content="x" * 900))
    snippet = rendered.split(": ", 1)[1]

    assert len(snippet) <= SNIPPET_MAX_CHARS
    assert snippet.endswith("\u2026")
    assert snippet.startswith("x" * 100)


def test_render_hit_keeps_a_snippet_exactly_at_the_cap_intact() -> None:
    content = "y" * SNIPPET_MAX_CHARS
    snippet = render_hit(hit(content=content)).split(": ", 1)[1]

    assert snippet == content
    assert "\u2026" not in snippet


@pytest.mark.parametrize("bad_type", [None, "", "   ", 42, {}])
def test_render_hit_falls_back_to_memory_for_a_missing_type(bad_type: Any) -> None:
    assert "[memory]" in render_hit(hit(type=bad_type))


def test_render_hit_falls_back_to_memory_when_type_is_absent() -> None:
    payload = hit()
    del payload["type"]
    assert "[memory]" in render_hit(payload)


@pytest.mark.parametrize("bad_title", [None, "", "   ", 7])
def test_render_hit_falls_back_to_untitled(bad_title: Any) -> None:
    assert "(untitled)" in render_hit(hit(title=bad_title))


def test_render_hit_falls_back_to_untitled_when_title_is_absent() -> None:
    payload = hit()
    del payload["title"]
    assert "(untitled)" in render_hit(payload)


@pytest.mark.parametrize(
    "bad_id",
    [None, "", "   ", "abc", "12x", True, False, 3.5, {}, []],
    ids=["none", "blank", "spaces", "letters", "mixed", "true", "false", "float", "dict", "list"],
)
def test_render_hit_skips_a_hit_with_no_usable_id(bad_id: Any) -> None:
    assert render_hit(hit(id=bad_id)) == ""


def test_render_hit_skips_a_hit_with_no_id_key_at_all() -> None:
    payload = hit()
    del payload["id"]
    assert render_hit(payload) == ""


def test_render_hit_accepts_a_numeric_string_id() -> None:
    assert render_hit(hit(id="761")).startswith("#761 ")


@pytest.mark.parametrize("empty", [None, "", "   ", "\n\n", 5, []])
def test_render_hit_omits_the_snippet_suffix_when_content_is_empty(empty: Any) -> None:
    rendered = render_hit(hit(id=9, type="bugfix", title="T", content=empty))
    assert rendered == "#9 [bugfix] T"
    assert not rendered.endswith(":")


def test_render_hit_is_exception_free_for_non_dict_input() -> None:
    for bad in (None, "x", 7, [], ()):
        assert render_hit(bad) == ""  # type: ignore[arg-type]


# -- 2. render_recall_block ---------------------------------------------------


def test_render_recall_block_renders_header_and_one_line_per_hit() -> None:
    hits = [hit(id=1, title="One"), hit(id=2, title="Two")]

    text, count = render_recall_block(hits, max_chars=2500)

    assert count == 2
    lines = text.splitlines()
    assert lines[0] == RECALL_HEADER
    assert lines[1] == ""
    assert lines[2].startswith("#1 ")
    assert lines[3].startswith("#2 ")


def test_render_recall_block_puts_the_context_section_before_the_hits() -> None:
    text, count = render_recall_block(
        [hit(id=1)], context="## Recent\n- did a thing", max_chars=2500
    )

    assert count == 1
    assert text.index("## Recent") < text.index("#1 ")
    assert text.startswith(RECALL_HEADER)


@pytest.mark.parametrize("empty_input", [[], (), None, "nonsense", 7, {}])
def test_render_recall_block_returns_nothing_for_no_input(empty_input: Any) -> None:
    assert render_recall_block(empty_input, max_chars=2500) == ("", 0)


def test_render_recall_block_returns_nothing_when_every_hit_is_unusable() -> None:
    assert render_recall_block([{"title": "no id"}, {}], max_chars=2500) == ("", 0)


@pytest.mark.parametrize("budget", [0, 1, 10, 26, 30, 40, 60, 100, 200, 400, 1200, 2500])
def test_render_recall_block_never_exceeds_the_budget(budget: int) -> None:
    hits = [hit(id=n, title=f"Title {n}", content="z" * 400) for n in range(1, 12)]

    text, count = render_recall_block(hits, context="c" * 900, max_chars=budget)

    assert len(text) <= budget, f"{len(text)} > {budget}"
    assert count >= 0


def test_render_recall_block_renders_nothing_when_not_even_the_header_fits() -> None:
    assert render_recall_block([hit()], max_chars=len(RECALL_HEADER) - 1) == ("", 0)


def test_render_recall_block_renders_nothing_when_only_the_header_would_fit() -> None:
    # A header-only block is not memory, but Hermes treats any non-empty return as
    # "memory was injected" and renders an indicator for it.
    text, count = render_recall_block([hit()], max_chars=len(RECALL_HEADER) + 2)

    assert (text, count) == ("", 0)


def test_render_recall_block_can_return_context_only_with_a_zero_count() -> None:
    long_hit = hit(id=1, title="T", content="z" * 400)
    budget = len(RECALL_HEADER) + 40

    text, count = render_recall_block([long_hit], context="c" * 200, max_chars=budget)

    assert count == 0
    assert text.startswith(RECALL_HEADER)
    assert "c" in text
    assert "#1 " not in text
    assert len(text) <= budget


def test_render_recall_block_truncates_a_huge_context_instead_of_dropping_every_hit() -> None:
    hits = [hit(id=1, title="One", content="short"), hit(id=2, title="Two", content="short")]

    text, count = render_recall_block(hits, context="c" * 5000, max_chars=400)

    assert len(text) <= 400
    assert count >= 1, "a context that blows the budget must be truncated, not win"
    assert "\u2026" in text
    assert "#1 " in text


def test_render_recall_block_drops_whole_hits_rather_than_truncating_one() -> None:
    hits = [hit(id=n, type="decision", title=f"Title {n}", content="payload") for n in (1, 2, 3, 4)]
    one_line = render_hit(hits[0])
    # Room for the header, the blank line and exactly two hit lines.
    budget = len(RECALL_HEADER) + 2 + len(one_line) + 1 + len(one_line) + 1

    text, count = render_recall_block(hits, max_chars=budget)

    assert count == 2
    assert render_hit(hits[0]) in text
    assert render_hit(hits[1]) in text
    assert render_hit(hits[2]) not in text
    # Nothing was cut mid-line: every rendered hit line is whole.
    for line in text.splitlines()[2:]:
        assert line in (render_hit(hits[0]), render_hit(hits[1]))


def test_render_recall_block_count_excludes_the_context_section() -> None:
    text, count = render_recall_block([hit(id=1)], context="project context", max_chars=2500)

    assert count == 1
    assert "project context" in text


def test_render_recall_block_skips_unusable_hits_without_counting_them() -> None:
    text, count = render_recall_block([{"title": "no id"}, hit(id=5)], max_chars=2500)

    assert count == 1
    assert "#5 " in text


def test_render_recall_block_emits_no_memory_context_tags() -> None:
    # Hermes wraps the result itself; a provider that pre-wraps gets stripped and
    # logged as a warning by sanitize_context.
    text, _ = render_recall_block([hit(id=1)], context="ctx", max_chars=2500)

    assert "<memory-context>" not in text
    assert "</memory-context>" not in text


# -- 3. render_status_line ----------------------------------------------------


def test_render_status_line_shows_unknown_values_as_a_dash() -> None:
    text = render_status_line("hermes-recall", server_ok=True)

    assert text.startswith("🧠 ")
    assert "hermes-recall" in text
    assert "reachable" in text
    # Every value the caller did not supply renders as a dash. Asserted per field
    # rather than by counting: the title line legitimately contains an em dash too.
    unknown_fields = [
        line for line in text.splitlines() if line.startswith("  ") and line.endswith("—")
    ]
    assert len(unknown_fields) == 4, text
    assert {line.split(":")[0].strip() for line in unknown_fields} == {
        "last error",
        "last memory",
        "last action",
        "last recall",
    }


def test_render_status_line_shows_every_known_value() -> None:
    text = render_status_line(
        "hermes-recall",
        server_ok=True,
        last_error="",
        last_memory_id=761,
        last_action="saved",
        recall_count=3,
        url="http://127.0.0.1:7437",
    )

    assert "hermes-recall" in text
    assert "http://127.0.0.1:7437 (reachable)" in text
    assert "#761" in text
    assert "saved" in text
    assert "3 memories" in text


def test_render_status_line_reports_an_unreachable_server_and_its_error() -> None:
    text = render_status_line(
        "", server_ok=False, last_error="GET /health: unreachable (connection refused)"
    )

    assert "unreachable" in text
    assert "connection refused" in text
    # An unknown project still renders, as a dash.
    assert "project:" in text
    assert "—" in text


def test_render_status_line_singularises_one_memory() -> None:
    assert "1 memory" in render_status_line("p", server_ok=True, recall_count=1)
    assert "0 memories" in render_status_line("p", server_ok=True, recall_count=0)


def test_render_status_line_is_deterministic() -> None:
    first = render_status_line("p", server_ok=True, last_memory_id=1)
    second = render_status_line("p", server_ok=True, last_memory_id=1)

    assert first == second


def test_render_status_line_uses_no_emoji_beyond_the_leading_glyph() -> None:
    text = render_status_line("p", server_ok=True, last_memory_id=1, recall_count=2)

    assert text.count("🧠") == 1
    assert text.startswith("🧠")


# -- 4. write confirmations ---------------------------------------------------


def test_saved_status() -> None:
    assert saved_status(761) == "🧠 saved #761"


def test_updated_status() -> None:
    assert updated_status(761) == "🧠 updated #761"
