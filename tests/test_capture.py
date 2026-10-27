"""Capture behaviour of ``RecallProvider``: prompts, passive turn capture, delegation.

Split out of ``tests/test_provider.py`` to keep both files readable — that one covers
availability, initialize, project detection, the system prompt and recall; this one
covers the three write hooks T4 added.

Nothing here touches the user's real Engram: every test points the provider at a
:class:`~fake_engram.FakeEngram` on an ephemeral port, or at a port with nothing
listening. Background work is always *drained*, never slept on.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fake_engram import FakeEngram, RecordedRequest

from hermes_recall import provider as provider_module
from hermes_recall.config import RecallConfig
from hermes_recall.provider import (
    DELEGATION_RESULT_MAX_CHARS,
    DELEGATION_TASK_MAX_CHARS,
    PASSIVE_MAX_CAPTURES_PER_TURN,
    PASSIVE_MIN_CHARS,
    PROMPT_MAX_CHARS,
    PROMPT_MIN_CHARS,
    RecallProvider,
)
from hermes_recall.redact import REDACTED

PROJECT = "hermes-recall"
SESSION = "sess-1"

# -- helpers ------------------------------------------------------------------


def make_config(url: str, **over: Any) -> RecallConfig:
    values: dict[str, Any] = {
        "url": url,
        "project_override": "",
        "recall_limit": 5,
        "recall_max_chars": 2500,
        "enabled": True,
        "auth_token": "",
    }
    values.update(over)
    return RecallConfig(**values)


def echo_session(request: RecordedRequest) -> tuple[int, Any]:
    """Acknowledge ``POST /sessions`` with the id it was sent, as Engram 2.0.0 does."""
    return (201, {"id": request.json_body["id"], "status": "created"})


def wire_engram(fake: FakeEngram, *, project: str = PROJECT) -> None:
    """Route the endpoints the capture paths touch."""
    fake.route("GET", "/project/current", (200, {"project": project}))
    fake.route("POST", "/sessions", echo_session)
    fake.route("POST", "/prompts", (201, {"id": 12, "status": "saved"}))
    fake.route(
        "POST", "/observations/passive", (200, {"extracted": 1, "saved": 1, "duplicates": 0})
    )


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def passive_requests(fake: FakeEngram) -> list[RecordedRequest]:
    """Every recorded passive capture, in arrival order."""
    return [r for r in fake.requests if r.method == "POST" and r.path == "/observations/passive"]


def captured(fake: FakeEngram) -> list[tuple[str, str]]:
    """Every passive capture as ``(source, content)``."""
    pairs = []
    for request in passive_requests(fake):
        body = request.json_body if isinstance(request.json_body, dict) else {}
        pairs.append((body.get("source", ""), body.get("content", "")))
    return pairs


def post_requests(fake: FakeEngram) -> list[RecordedRequest]:
    return [r for r in fake.requests if r.method == "POST"]


def tool_message(content: str, *, name: str = "", call_id: str = "") -> dict[str, Any]:
    """One OpenAI-style tool result message."""
    message: dict[str, Any] = {"role": "tool", "content": content}
    if name:
        message["name"] = name
    if call_id:
        message["tool_call_id"] = call_id
    return message


def assistant_with_calls(*calls: tuple[str, str]) -> dict[str, Any]:
    """An assistant message issuing ``(call_id, function_name)`` tool calls."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
            for call_id, name in calls
        ],
    }


def long_text(marker: str = "x") -> str:
    """Content comfortably over ``PASSIVE_MIN_CHARS``."""
    return marker * (PASSIVE_MIN_CHARS + 20)


@pytest.fixture
def provider_factory() -> Iterator[Any]:
    """Build providers and shut their workers down on teardown."""
    built: list[RecallProvider] = []

    def build(
        url: str, *, session_id: str = SESSION, agent_context: str = "primary", **config_over: Any
    ) -> RecallProvider:
        provider = RecallProvider(config=make_config(url, **config_over))
        built.append(provider)
        # Capture needs a registered session id and a writes verdict, both of which
        # initialize() sets. Set them directly so each test controls exactly one
        # variable; initialize()'s own behaviour is covered in test_provider.py.
        provider.initialize(
            session_id=session_id,
            hermes_home="/tmp/hermes",
            platform="cli",
            agent_context=agent_context,
        )
        assert provider._drain_for_test()
        return provider

    try:
        yield build
    finally:
        for provider in built:
            provider.shutdown()


# -- 1. on_turn_start: the prompt log -----------------------------------------


def test_on_turn_start_posts_the_prompt(fake_engram: FakeEngram, provider_factory: Any) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, "why does the loader skip my provider")
    assert provider._drain_for_test()

    request = fake_engram.find("POST", "/prompts")
    assert request is not None
    assert request.json_body == {
        "session_id": SESSION,
        "content": "why does the loader skip my provider",
        "project": PROJECT,
        # No project_source in the detection answer: the evidence is unknown.
        "project_provenance": {
            "source": "unknown",
            "directory": provider_module._working_directory(),
        },
    }


def test_on_turn_start_tracks_the_turn_number(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.on_turn_start(7, "a real question about the loader")
    assert provider._drain_for_test()

    assert provider._turn_number == 7


def test_on_turn_start_redacts_private_blocks(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, "deploy with <private>hunter2</private> please")
    assert provider._drain_for_test()

    request = fake_engram.find("POST", "/prompts")
    assert request is not None
    assert request.json_body["content"] == f"deploy with {REDACTED} please"
    assert "hunter2" not in request.json_body["content"]


def test_on_turn_start_truncates_at_the_prompt_cap(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, "q" * 5000)
    assert provider._drain_for_test()

    request = fake_engram.find("POST", "/prompts")
    assert request is not None
    content = request.json_body["content"]
    assert content == "q" * PROMPT_MAX_CHARS + "..."
    assert len(content) == PROMPT_MAX_CHARS + 3


def test_on_turn_start_does_not_leak_a_block_straddling_the_cap(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The deviation from upstream, asserted as the SAFE behaviour: upstream truncates
    # before stripping, so a <private> block cut by the 2000-char boundary loses its
    # closing tag, matches nothing, and leaks. We redact first, so it cannot.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    secret = "s" * 400
    message = "a" * (PROMPT_MAX_CHARS - 100) + f"<private>{secret}</private>" + "b" * 500

    provider.on_turn_start(1, message)
    assert provider._drain_for_test()

    request = fake_engram.find("POST", "/prompts")
    assert request is not None
    content = request.json_body["content"]
    assert "s" * 20 not in content, "the straddling private block leaked"
    assert REDACTED in content
    assert len(content) <= PROMPT_MAX_CHARS + 3


def test_on_turn_start_does_not_skip_a_trivial_prompt(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Deliberate asymmetry with prefetch: a prompt is the user's own words, and the
    # prompt log is a different artifact from recall. "go ahead please" is worthless
    # as a search query and still worth having in the record of what was asked.
    from hermes_recall._hermes import is_trivial_prompt

    # "thank you" plus trailing punctuation: trivial for recall, and 12 chars, so it
    # clears the prompt floor. (A bare "ok" would be skipped for length, not triviality,
    # which would prove nothing.)
    trivial = "thank you!!!"
    assert is_trivial_prompt(trivial), "test premise: this must be trivial for prefetch"
    assert len(trivial) > PROMPT_MIN_CHARS, "test premise: it must clear the length floor"

    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, trivial)
    assert provider._drain_for_test()

    request = fake_engram.find("POST", "/prompts")
    assert request is not None
    assert request.json_body["content"] == trivial


@pytest.mark.parametrize(
    "message",
    ["", "   ", "short", "x" * PROMPT_MIN_CHARS, "<private>hidden</private>"],
    ids=["empty", "blank", "short", "exactly-at-the-floor", "fully-redacted-to-a-marker"],
)
def test_on_turn_start_skips_content_that_does_not_clear_the_floor(
    fake_engram: FakeEngram, provider_factory: Any, message: str
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, message)
    assert provider._drain_for_test()

    assert fake_engram.find("POST", "/prompts") is None


def test_on_turn_start_posts_one_char_above_the_floor(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Upstream's rule is strictly greater than 10, so the boundary is pinned on both
    # sides: the case above proves == is skipped, this proves +1 is kept.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, "x" * (PROMPT_MIN_CHARS + 1))
    assert provider._drain_for_test()

    assert fake_engram.find("POST", "/prompts") is not None


def test_on_turn_start_makes_no_request_when_disabled(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, enabled=False)
    fake_engram.reset()

    provider.on_turn_start(1, "a real question about the loader")
    assert provider._drain_for_test()

    assert fake_engram.requests == []


def test_on_turn_start_makes_no_request_without_a_session_id(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, session_id="")
    fake_engram.reset()

    provider.on_turn_start(1, "a real question about the loader")
    assert provider._drain_for_test()

    assert fake_engram.requests == []


def test_on_turn_start_drops_the_prompt_when_the_project_is_unknown(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, project="")
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, "a real question about the loader")
    assert provider._drain_for_test()

    assert fake_engram.find("POST", "/prompts") is None


def test_on_turn_start_uses_the_host_skill_extractor_when_present(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The host does NOT strip skill scaffolding on the on_turn_start fan-out (it does
    # for prefetch_all and sync_all), so a /skill turn arrives expanded. We borrow the
    # host's own extractor rather than guess the format.
    monkeypatch.setattr(
        provider_module, "_skill_extractor", lambda _message: "the real instruction here"
    )
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, "<skill>a thousand lines of skill body</skill> do the thing")
    assert provider._drain_for_test()

    request = fake_engram.find("POST", "/prompts")
    assert request is not None
    assert request.json_body["content"] == "the real instruction here"


def test_on_turn_start_skips_a_bare_skill_invocation(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The host's extractor returns None for a skill invocation with no user
    # instruction, which means "there is nothing the user actually asked".
    monkeypatch.setattr(provider_module, "_skill_extractor", lambda _message: None)
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, "<skill>a thousand lines of skill body</skill>")
    assert provider._drain_for_test()

    assert fake_engram.find("POST", "/prompts") is None


def test_on_turn_start_falls_back_when_the_extractor_raises(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_message: str) -> str:
        raise RuntimeError("host helper changed shape")

    monkeypatch.setattr(provider_module, "_skill_extractor", boom)
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, "a real question about the loader")
    assert provider._drain_for_test()

    request = fake_engram.find("POST", "/prompts")
    assert request is not None
    assert request.json_body["content"] == "a real question about the loader"


def test_the_skill_extractor_lookup_caches_its_miss() -> None:
    # Python does not cache a failed import, so an unresolved lookup would re-scan
    # sys.path on every turn — on the prompt path, before prefetch.
    assert provider_module._skill_instruction_extractor() is None
    assert provider_module._skill_extractor is None
    assert provider_module._skill_extractor is not provider_module._EXTRACTOR_UNRESOLVED


# -- 2. sync_turn: passive capture --------------------------------------------


def test_sync_turn_captures_tool_results_and_the_assistant_text(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [
        {"role": "user", "content": "do the thing"},
        assistant_with_calls(("call-1", "read_file")),
        tool_message(long_text("r"), call_id="call-1"),
        {"role": "assistant", "content": long_text("a")},
    ]

    provider.sync_turn("do the thing", long_text("a"), session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    # The assistant reply comes first so it survives the per-turn cap.
    assert captured(fake_engram) == [
        ("assistant", long_text("a")),
        ("read_file", long_text("r")),
    ]
    body = passive_requests(fake_engram)[0].json_body
    assert body["session_id"] == SESSION
    assert body["project"] == PROJECT


def test_sync_turn_returns_before_the_posts_land(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [tool_message(long_text("r"), name="read_file")]

    started = time.monotonic()
    provider.sync_turn("ask", long_text("a"), session_id=SESSION, messages=messages)
    elapsed = time.monotonic() - started

    # Non-blocking: the call returns promptly and the work shows up only after a drain.
    assert elapsed < 0.5, f"sync_turn blocked for {elapsed:.3f}s"
    assert provider._drain_for_test()
    assert len(passive_requests(fake_engram)) == 2


def test_sync_turn_derives_the_source_from_the_message_name(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [tool_message(long_text("r"), name="run_command", call_id="call-9")]

    provider.sync_turn("ask", "", session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    assert captured(fake_engram) == [("run_command", long_text("r"))]


def test_sync_turn_falls_back_to_the_tool_call_function_name(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [
        assistant_with_calls(("call-7", "web_search")),
        tool_message(long_text("r"), call_id="call-7"),
    ]

    provider.sync_turn("ask", "", session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    assert captured(fake_engram) == [("web_search", long_text("r"))]


def test_sync_turn_falls_back_to_the_literal_tool_source(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [tool_message(long_text("r"))]

    provider.sync_turn("ask", "", session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    assert captured(fake_engram) == [("tool", long_text("r"))]


def test_sync_turn_skips_this_providers_own_tool_results(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Capturing a mem_search result would write recalled memories back in as new
    # observations — a feedback loop that grows the store every time the model looks
    # something up. Upstream guards the same way.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [
        tool_message(long_text("m"), name="mem_search"),
        tool_message(long_text("s"), name="mem_save"),
        tool_message(long_text("r"), name="read_file"),
    ]

    provider.sync_turn("ask", "", session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    assert captured(fake_engram) == [("read_file", long_text("r"))]


@pytest.mark.parametrize(
    ("length", "expected"),
    [(PASSIVE_MIN_CHARS, 0), (PASSIVE_MIN_CHARS + 1, 1)],
    ids=["exactly-50-is-skipped", "51-is-captured"],
)
def test_sync_turn_honours_the_strict_length_floor(
    fake_engram: FakeEngram, provider_factory: Any, length: int, expected: int
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [tool_message("t" * length, name="read_file")]

    provider.sync_turn("ask", "", session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    assert len(passive_requests(fake_engram)) == expected


def test_sync_turn_measures_the_floor_before_redaction(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Upstream checks content.length <= 50 on the RAW result, so a long block that
    # redacts down to little is still offered to Engram's extractor.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    raw = "keep " + "<private>" + "s" * 200 + "</private>"
    assert len(raw) > PASSIVE_MIN_CHARS

    provider.sync_turn("ask", "", session_id=SESSION, messages=[tool_message(raw, name="r")])
    assert provider._drain_for_test()

    assert captured(fake_engram) == [("r", f"keep {REDACTED}")]


def test_sync_turn_captures_each_tool_result_exactly_once(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The high-water mark. The host hands us the WHOLE conversation every turn, so
    # without it every turn would re-post every earlier turn's tool results.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    turn_one = [
        {"role": "user", "content": "first"},
        tool_message(long_text("one"), name="read_file"),
    ]
    provider.sync_turn("first", "", session_id=SESSION, messages=turn_one)
    assert provider._drain_for_test()
    assert captured(fake_engram) == [("read_file", long_text("one"))]

    fake_engram.reset()
    turn_two = turn_one + [
        {"role": "user", "content": "second"},
        tool_message(long_text("two"), name="write_file"),
    ]
    provider.sync_turn("second", "", session_id=SESSION, messages=turn_two)
    assert provider._drain_for_test()

    # Only the new result, and the first one is not repeated.
    assert captured(fake_engram) == [("write_file", long_text("two"))]


def test_sync_turn_handles_a_transcript_that_shrank(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Compaction or a rewind can shorten the list under us. Treat everything present
    # as already seen rather than re-post a whole history; T5 owns the explicit hooks.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    long_history = [tool_message(long_text(str(n)), name="read_file") for n in range(6)]
    provider.sync_turn("ask", "", session_id=SESSION, messages=long_history)
    assert provider._drain_for_test()
    fake_engram.reset()

    provider.sync_turn("ask", "", session_id=SESSION, messages=long_history[:2])
    assert provider._drain_for_test()

    assert passive_requests(fake_engram) == []
    assert provider._synced_message_count == 2


def test_sync_turn_with_no_messages_still_captures_the_assistant_text(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.sync_turn("ask", long_text("a"), session_id=SESSION, messages=None)
    assert provider._drain_for_test()

    assert captured(fake_engram) == [("assistant", long_text("a"))]


def test_sync_turn_with_no_messages_leaves_the_high_water_mark_alone(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.sync_turn("ask", "", session_id=SESSION, messages=[tool_message(long_text("r"))])
    assert provider._drain_for_test()
    mark = provider._synced_message_count

    provider.sync_turn("ask", long_text("a"), session_id=SESSION, messages=None)
    assert provider._drain_for_test()

    assert provider._synced_message_count == mark


def test_sync_turn_stops_at_the_per_turn_capture_cap(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [
        tool_message(long_text(f"t{n}"), name="read_file")
        for n in range(PASSIVE_MAX_CAPTURES_PER_TURN + 10)
    ]

    provider.sync_turn("ask", long_text("a"), session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    assert len(passive_requests(fake_engram)) == PASSIVE_MAX_CAPTURES_PER_TURN
    # The assistant reply is first in line, so the cap never starves it.
    assert captured(fake_engram)[0] == ("assistant", long_text("a"))


def test_sync_turn_skips_unrecognised_message_shapes(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages: list[Any] = [
        "not a dict",
        None,
        {"role": "tool"},
        {"role": "tool", "content": None},
        {"role": "tool", "content": [{"type": "text", "text": long_text("p")}]},
        {"role": "user", "content": long_text("u")},
        tool_message(long_text("r"), name="read_file"),
    ]

    provider.sync_turn("ask", "", session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    assert captured(fake_engram) == [("read_file", long_text("r"))]


def test_one_failing_capture_does_not_abandon_the_others(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    answers: list[Any] = [
        (500, {"error": "boom"}),
        (200, {"saved": 1}),
        (200, {"saved": 1}),
    ]

    def answer(_request: RecordedRequest) -> Any:
        return answers.pop(0) if answers else (200, {"saved": 1})

    fake_engram.route("POST", "/observations/passive", answer)
    messages = [
        tool_message(long_text("one"), name="read_file"),
        tool_message(long_text("two"), name="write_file"),
    ]

    provider.sync_turn("ask", long_text("a"), session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    # All three were attempted; the 500 on the first did not abort the turn.
    assert [source for source, _ in captured(fake_engram)] == [
        "assistant",
        "read_file",
        "write_file",
    ]


def test_sync_turn_makes_no_request_when_disabled(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, enabled=False)
    fake_engram.reset()

    provider.sync_turn("ask", long_text("a"), session_id=SESSION, messages=None)
    assert provider._drain_for_test()

    assert fake_engram.requests == []


def test_sync_turn_drops_captures_when_the_project_is_unknown(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, project="")
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.sync_turn("ask", long_text("a"), session_id=SESSION, messages=None)
    assert provider._drain_for_test()

    assert passive_requests(fake_engram) == []


def test_sync_turn_prefers_an_explicit_session_id(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.sync_turn("ask", long_text("a"), session_id="gateway-session", messages=None)
    assert provider._drain_for_test()

    request = passive_requests(fake_engram)[0]
    assert request.json_body["session_id"] == "gateway-session"


def test_sync_turn_declares_messages_so_the_host_passes_it() -> None:
    # The host probes the signature and only passes `messages` when the parameter is
    # declared (_provider_sync_accepts_messages). Dropping it would silently turn off
    # tool-result capture.
    import inspect

    assert "messages" in inspect.signature(RecallProvider.sync_turn).parameters


# -- 3. on_delegation ---------------------------------------------------------


def test_on_delegation_captures_task_and_result(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_delegation(
        "research the loader",
        "the loader skips providers whose prefetch is still running",
        child_session_id="child-1",
    )
    assert provider._drain_for_test()

    requests = passive_requests(fake_engram)
    assert len(requests) == 1
    body = requests[0].json_body
    assert body["source"] == "subagent"
    assert body["session_id"] == SESSION
    assert body["project"] == PROJECT
    assert "research the loader" in body["content"]
    assert "prefetch is still running" in body["content"]


def test_on_delegation_keeps_the_child_session_id_out_of_the_content(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The subagent has no provider session of its own, so the id refers to nothing
    # retrievable from Engram and would be an unresolvable reference in the record.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_delegation("task text here", "result text here", child_session_id="child-xyz")
    assert provider._drain_for_test()

    assert "child-xyz" not in passive_requests(fake_engram)[0].json_body["content"]


def test_on_delegation_truncates_both_halves(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_delegation("t" * 9000, "r" * 90000)
    assert provider._drain_for_test()

    content = passive_requests(fake_engram)[0].json_body["content"]
    assert "t" * DELEGATION_TASK_MAX_CHARS + "..." in content
    assert "r" * DELEGATION_RESULT_MAX_CHARS + "..." in content
    assert len(content) < 9000 + 90000


def test_on_delegation_redacts_private_blocks(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_delegation("look at <private>hunter2</private> config", "found the issue here")
    assert provider._drain_for_test()

    content = passive_requests(fake_engram)[0].json_body["content"]
    assert "hunter2" not in content
    assert REDACTED in content


def test_on_delegation_skips_a_tiny_delegation(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    # The composed block carries fixed labels, so an empty pair is still a few dozen
    # characters; the floor is measured on the composed content.
    provider.on_delegation("", "")
    assert provider._drain_for_test()

    composed = provider_module._compose_delegation("", "")
    expected = 1 if len(composed) > PASSIVE_MIN_CHARS else 0
    assert len(passive_requests(fake_engram)) == expected


def test_on_delegation_makes_no_request_when_disabled(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, enabled=False)
    fake_engram.reset()

    provider.on_delegation("a real task description", "a real result description")
    assert provider._drain_for_test()

    assert fake_engram.requests == []


# -- 4. the non-primary write gate --------------------------------------------


@pytest.mark.parametrize("context", ["subagent", "cron", "flush"])
def test_no_capture_path_writes_for_a_non_primary_context(
    fake_engram: FakeEngram, provider_factory: Any, context: str
) -> None:
    # The whole point of the gate: a cron system prompt or a subagent transcript would
    # land in the user's project memory as if the user had said it.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, agent_context=context)
    messages = [tool_message(long_text("r"), name="read_file")]

    provider.on_turn_start(1, "a real question about the loader")
    provider.sync_turn("ask", long_text("a"), session_id=SESSION, messages=messages)
    provider.on_delegation("a real task description", "a real result description")
    assert provider._drain_for_test()

    assert provider._writes_enabled is False
    assert post_requests(fake_engram) == []


# -- 5. Engram down -----------------------------------------------------------


def test_every_capture_hook_survives_an_absent_server(provider_factory: Any) -> None:
    provider = provider_factory(f"http://127.0.0.1:{unused_port()}")
    messages = [tool_message(long_text("r"), name="read_file")]

    provider.on_turn_start(1, "a real question about the loader")
    provider.sync_turn("ask", long_text("a"), session_id=SESSION, messages=messages)
    provider.on_delegation("a real task description", "a real result description")

    # The worker drained rather than wedging, and shutdown still completes.
    assert provider._drain_for_test(timeout=10.0)
    provider.shutdown()
    assert provider._worker is None


def test_capture_hooks_never_raise_on_garbage_input(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.on_turn_start("not an int", None)  # type: ignore[arg-type]
    provider.sync_turn(None, None, session_id=None, messages="not a list")  # type: ignore[arg-type]
    provider.on_delegation(None, None, child_session_id=None)  # type: ignore[arg-type]

    assert provider._drain_for_test()


# -- 6. redaction runs before every truncation ----------------------------------

SECRET = "SECRETZ"


def private_straddling(cut: int, *, filler: str = "a") -> str:
    """Text whose ``<private>`` block opens just before ``cut`` and closes well after it."""
    return filler * (cut - 20) + "<private>" + SECRET * 100 + "</private>" + "b" * 300


@pytest.mark.parametrize(
    ("task", "result"),
    [
        (private_straddling(DELEGATION_TASK_MAX_CHARS), "found the issue in the loader"),
        ("look at the loader please", private_straddling(DELEGATION_RESULT_MAX_CHARS)),
    ],
    ids=["task-cut", "result-cut"],
)
def test_on_delegation_does_not_leak_a_block_straddling_a_cap(
    fake_engram: FakeEngram, provider_factory: Any, task: str, result: str
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_delegation(task, result)
    assert provider._drain_for_test()

    content = passive_requests(fake_engram)[0].json_body["content"]
    assert "SECRET" not in content, "the straddling private block leaked"
    assert REDACTED in content


# -- 7. session registration is retried when initialize could not register ------


def test_a_session_unregistered_at_initialize_registers_on_the_next_write(
    provider_factory: Any,
) -> None:
    # Engram is down at initialize, so POST /sessions never lands; it comes up later.
    port = unused_port()
    provider = provider_factory(f"http://127.0.0.1:{port}")
    with FakeEngram(port=port) as fake:
        wire_engram(fake)
        registered: set[str] = set()

        def sessions(request: RecordedRequest) -> tuple[int, Any]:
            registered.add(request.json_body["id"])
            return (201, {"id": request.json_body["id"], "status": "created"})

        def needs_session(status: int, payload: Any) -> Any:
            def route(request: RecordedRequest) -> tuple[int, Any]:
                if request.json_body["session_id"] not in registered:
                    return (404, {"error": "session not found"})
                return (status, payload)

            return route

        fake.route("POST", "/sessions", sessions)
        fake.route("POST", "/prompts", needs_session(201, {"id": 12, "status": "saved"}))
        fake.route(
            "POST",
            "/observations/passive",
            needs_session(200, {"extracted": 1, "saved": 1, "duplicates": 0}),
        )

        provider.on_turn_start(1, "why does the loader skip my provider")
        provider.sync_turn(
            "ask", "", session_id=SESSION, messages=[tool_message(long_text("r"), name="read")]
        )
        assert provider._drain_for_test()

        prompts = [r for r in fake.requests if r.path == "/prompts"]
        assert [r.json_body["content"] for r in prompts] == ["why does the loader skip my provider"]
        assert captured(fake) == [("read", long_text("r"))]
        assert sum(1 for r in fake.requests if r.path == "/sessions") == 1


def test_a_session_not_found_write_registers_and_retries_exactly_once(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    fake_engram.route("POST", "/prompts", (404, {"error": "session not found"}))
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_turn_start(1, "why does the loader skip my provider")
    assert provider._drain_for_test()

    paths = [r.path for r in fake_engram.requests if r.method == "POST"]
    assert paths == ["/prompts", "/sessions", "/prompts"], paths


def test_a_session_not_found_write_lands_after_the_retry(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    attempts: list[int] = []

    def passive(_request: RecordedRequest) -> tuple[int, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            return (404, {"error": "session not found"})
        return (200, {"extracted": 1, "saved": 1, "duplicates": 0})

    fake_engram.route("POST", "/observations/passive", passive)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_delegation("look at the loader please", long_text("r"))
    assert provider._drain_for_test()

    assert len(attempts) == 2
    assert sum(1 for r in fake_engram.requests if r.path == "/sessions") == 1


# -- 8. MCP Engram tool results are never passively captured --------------------


@pytest.mark.parametrize("name", ["mcp__engram__mem_search", "mcp_engram_mem_save"])
def test_sync_turn_skips_mcp_engram_tool_results(
    fake_engram: FakeEngram, provider_factory: Any, name: str
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [
        tool_message(long_text("m"), name=name),
        tool_message(long_text("r"), name="read_file"),
    ]

    provider.sync_turn("ask", "", session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    assert captured(fake_engram) == [("read_file", long_text("r"))]


def test_the_first_sync_after_startup_skips_a_resumed_transcript(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Starting Hermes on a resumed session hands the first sync_turn the whole old
    # conversation; its tool results were captured by the run that produced them.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [
        {"role": "user", "content": "an old question"},
        tool_message(long_text("o"), name="read_file"),
        {"role": "user", "content": "the new question"},
        tool_message(long_text("n"), name="read_file"),
    ]

    provider.sync_turn("ask", "", session_id=SESSION, messages=messages)
    assert provider._drain_for_test()

    assert captured(fake_engram) == [("read_file", long_text("n"))]


# -- capture project provenance (Engram 3.6, S6) --------------------------------

WORKDIR = "/work/hermes-recall"
PROVENANCE_ROUTES = ("/sessions", "/prompts", "/observations", "/observations/passive")


def wire_provenance(fake: FakeEngram, source: Any) -> None:
    wire_engram(fake)
    detected: dict[str, Any] = {"project": PROJECT}
    if source is not None:
        detected["project_source"] = source
    fake.route("GET", "/project/current", (200, detected))
    fake.route("POST", "/observations", (201, {"id": 900, "status": "saved"}))
    fake.route(
        "POST",
        f"/sessions/{SESSION}/finalize",
        (200, {"session_id": SESSION, "state": "finalized", "summary_id": 901}),
    )


def every_capture_write(provider: RecallProvider) -> None:
    """Drive each write path that carries provenance, then drain."""
    provider.on_turn_start(1, "why does the loader skip my provider")
    provider.sync_turn("ask", long_text("a"), messages=[tool_message(long_text("t"), name="read")])
    provider.on_delegation("look at the loader please", long_text("d"))
    assert provider._drain_for_test()
    provider.handle_tool_call("mem_save", {"title": "T", "content": "C"})
    provider.handle_tool_call("mem_session_summary", {"content": "## Goal\nship"})
    provider.on_pre_compress(
        [{"role": "user", "content": "a question"}, {"role": "assistant", "content": "an answer"}]
    )


def provenances(fake: FakeEngram) -> list[tuple[str, Any]]:
    """``(path, project_provenance)`` for every capture write, in order."""
    return [
        (r.path, r.json_body.get("project_provenance"))
        for r in fake.requests
        if r.method == "POST" and r.path in PROVENANCE_ROUTES
    ]


def test_every_capture_write_carries_the_detected_provenance(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERMINAL_CWD", WORKDIR)
    wire_provenance(fake_engram, "git_remote")
    provider = provider_factory(fake_engram.base_url)

    every_capture_write(provider)

    expected = {"source": "git_remote", "directory": WORKDIR}
    assert provenances(fake_engram) == [
        ("/sessions", expected),
        ("/prompts", expected),
        ("/observations/passive", expected),
        ("/observations/passive", expected),
        ("/observations/passive", expected),
        ("/observations", expected),
        ("/observations", expected),
        ("/observations", expected),
    ]


def test_finalize_never_carries_provenance(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Its body is strict: {project, summary} and nothing else.
    monkeypatch.setenv("TERMINAL_CWD", WORKDIR)
    wire_provenance(fake_engram, "git_remote")
    provider = provider_factory(fake_engram.base_url)

    provider.handle_tool_call("mem_session_finalize", {"summary": "done"})

    request = fake_engram.find("POST", f"/sessions/{SESSION}/finalize")
    assert request is not None
    assert request.json_body == {"project": PROJECT, "summary": "done"}


@pytest.mark.parametrize("source", ["cwd", "", None, "made_up_source", 7])
def test_a_source_outside_engrams_set_is_sent_as_unknown(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch, source: Any
) -> None:
    # Engram 400s an unsupported source, which would lose the whole write.
    monkeypatch.setenv("TERMINAL_CWD", WORKDIR)
    wire_provenance(fake_engram, source)
    provider = provider_factory(fake_engram.base_url)

    every_capture_write(provider)

    rows = provenances(fake_engram)
    assert len(rows) == 8
    assert {path for path, _ in rows} == set(PROVENANCE_ROUTES)
    assert all(p == {"source": "unknown", "directory": WORKDIR} for _, p in rows)


def test_a_configured_project_is_sent_as_config(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERMINAL_CWD", WORKDIR)
    wire_provenance(fake_engram, "git_remote")
    provider = provider_factory(fake_engram.base_url, project_override=PROJECT)

    every_capture_write(provider)

    rows = provenances(fake_engram)
    assert len(rows) == 8
    assert all(p == {"source": "config", "directory": WORKDIR} for _, p in rows)
    assert fake_engram.find("GET", "/project/current") is None


def test_a_write_to_another_project_is_sent_as_unknown(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The detection evidence describes the detected project only.
    monkeypatch.setenv("TERMINAL_CWD", WORKDIR)
    wire_provenance(fake_engram, "git_remote")
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.handle_tool_call("mem_save", {"title": "T", "content": "C", "project": "other"})
    provider.handle_tool_call("mem_save", {"title": "T", "content": "C"})

    assert provenances(fake_engram) == [
        ("/observations", {"source": "unknown", "directory": WORKDIR}),
        ("/observations", {"source": "git_remote", "directory": WORKDIR}),
    ]
