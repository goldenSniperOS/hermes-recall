"""Lifecycle behaviour of ``RecallProvider``: session end, session switch, compaction.

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
from fake_engram import Delay, FakeEngram, RecordedRequest

from hermes_recall import client as client_module
from hermes_recall import provider as provider_module
from hermes_recall.config import RecallConfig
from hermes_recall.provider import (
    _ENDED_SESSION_MEMORY,
    PRE_COMPRESS_MAX_CHARS,
    RECOVERY_MAX_CHARS,
    RecallProvider,
)
from hermes_recall.redact import REDACTED

PROJECT = "hermes-recall"
SESSION = "sess-1"
NEXT_SESSION = "sess-2"

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


def wire_engram(
    fake: FakeEngram,
    *,
    project: str = PROJECT,
    compaction_context: str = "## Earlier\n- shipped the client",
) -> None:
    """Route every endpoint the lifecycle paths touch."""
    fake.route("GET", "/project/current", (200, {"project": project}))
    hit = {"id": 761, "type": "decision", "title": "T", "content": "c"}
    fake.route("GET", "/search", (200, [hit]))
    fake.route("GET", "/context", (200, {"context": "## Recent\n- did a thing"}))
    fake.route("GET", "/context/compaction", (200, {"context": compaction_context}))
    fake.route("POST", "/sessions", (201, {"id": SESSION, "status": "created"}))
    fake.route("POST", "/observations", (201, {"id": 900, "status": "saved"}))
    fake.route(
        "POST", "/observations/passive", (200, {"extracted": 1, "saved": 1, "duplicates": 0})
    )
    # Any /sessions/<id>/end path. FakeEngram routes on the exact path, so the default
    # answers the ones we do not name explicitly.
    fake.default_response = (200, {"id": "", "status": "completed"})


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def end_requests(fake: FakeEngram) -> list[RecordedRequest]:
    """Every recorded session-end POST, in arrival order."""
    return [
        r
        for r in fake.requests
        if r.method == "POST" and r.path.startswith("/sessions/") and r.path.endswith("/end")
    ]


def count_requests(fake: FakeEngram, method: str, path: str) -> int:
    return sum(1 for r in fake.requests if r.method == method.upper() and r.path == path)


def observation_requests(fake: FakeEngram) -> list[RecordedRequest]:
    return [r for r in fake.requests if r.method == "POST" and r.path == "/observations"]


def tool_message(content: str, *, name: str = "read_file") -> dict[str, Any]:
    return {"role": "tool", "content": content, "name": name}


def transcript(turns: int = 3) -> list[dict[str, Any]]:
    """A plausible conversation to digest."""
    rows: list[dict[str, Any]] = []
    for n in range(turns):
        rows.append({"role": "user", "content": f"question number {n} about the loader"})
        rows.append({"role": "assistant", "content": f"answer number {n} with some detail"})
    return rows


@pytest.fixture
def provider_factory() -> Iterator[Any]:
    """Build initialized providers and shut their workers down on teardown."""
    built: list[RecallProvider] = []

    def build(
        url: str, *, session_id: str = SESSION, agent_context: str = "primary", **config_over: Any
    ) -> RecallProvider:
        provider = RecallProvider(config=make_config(url, **config_over))
        built.append(provider)
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


# -- 1. on_session_end --------------------------------------------------------


def test_on_session_end_posts_an_empty_summary(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Deliberately empty: the meaningful end-of-session artifact is the model-authored
    # mem_session_summary, and upstream gentle-engram sends {"summary": ""} too.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_session_end(transcript())
    assert provider._drain_for_test()

    requests = end_requests(fake_engram)
    assert len(requests) == 1
    assert requests[0].path == f"/sessions/{SESSION}/end"
    assert requests[0].json_body == {"summary": ""}


def test_on_session_end_records_the_session_as_ended(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.on_session_end([])
    assert provider._drain_for_test()

    assert SESSION in provider._ended_sessions


def test_on_session_end_makes_no_request_when_disabled(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, enabled=False)
    fake_engram.reset()

    provider.on_session_end(transcript())
    assert provider._drain_for_test()

    assert fake_engram.requests == []


@pytest.mark.parametrize("context", ["subagent", "cron", "flush"])
def test_on_session_end_makes_no_request_for_a_non_primary_context(
    fake_engram: FakeEngram, provider_factory: Any, context: str
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, agent_context=context)
    fake_engram.reset()

    provider.on_session_end(transcript())
    assert provider._drain_for_test()

    assert end_requests(fake_engram) == []


def test_on_session_end_makes_no_request_without_a_session_id(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, session_id="")
    fake_engram.reset()

    provider.on_session_end(transcript())
    assert provider._drain_for_test()

    assert fake_engram.requests == []


# -- 2. on_session_switch -----------------------------------------------------


def test_switch_with_reset_ends_the_old_session_and_registers_the_new_one(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_session_switch(NEXT_SESSION, parent_session_id=SESSION, reset=True)
    assert provider._drain_for_test()

    ended = end_requests(fake_engram)
    assert len(ended) == 1
    assert ended[0].path == f"/sessions/{SESSION}/end"
    assert ended[0].json_body == {"summary": ""}

    registered = fake_engram.find("POST", "/sessions")
    assert registered is not None
    assert registered.json_body["id"] == NEXT_SESSION
    assert registered.json_body["project"] == PROJECT
    assert provider._session_id == NEXT_SESSION


def test_switch_without_reset_registers_but_does_not_end(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # A /resume or /branch: the logical conversation continues under a new id, so the
    # old record must stay open.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_session_switch(NEXT_SESSION, parent_session_id=SESSION, reset=False)
    assert provider._drain_for_test()

    assert end_requests(fake_engram) == []
    registered = fake_engram.find("POST", "/sessions")
    assert registered is not None
    assert registered.json_body["id"] == NEXT_SESSION


def test_the_hosts_rotation_order_ends_the_session_exactly_once(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The de-duplication test. On the rotation path MemoryManager submits
    # on_session_end and then on_session_switch(reset=True) as ONE background task, in
    # that exact order. Without the check, every /reset would post /end twice.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_session_end(transcript())
    provider.on_session_switch(NEXT_SESSION, parent_session_id=SESSION, reset=True, reason="/reset")
    assert provider._drain_for_test()

    assert len(end_requests(fake_engram)) == 1


def test_switch_tolerates_the_hosts_extra_kwargs(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The host forwards `reason` on the rotation path and `rewound` only when true, so
    # nothing may depend on an exact kwargs shape.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.on_session_switch(
        NEXT_SESSION, parent_session_id=SESSION, reset=True, reason="/new", future_kwarg=1
    )
    assert provider._drain_for_test()

    assert provider._session_id == NEXT_SESSION


def test_switch_re_detects_the_project(fake_engram: FakeEngram, provider_factory: Any) -> None:
    # A /resume or /branch can land in a different working directory, and the project
    # is per-directory.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    assert count_requests(fake_engram, "GET", "/project/current") == 1

    provider.on_session_switch(NEXT_SESSION, reset=True)
    assert provider._drain_for_test()

    assert count_requests(fake_engram, "GET", "/project/current") == 2


def test_switch_makes_the_next_prefetch_fetch_the_context_again(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.prefetch("a real question about the loader")
    assert count_requests(fake_engram, "GET", "/context") == 1

    provider.on_session_switch(NEXT_SESSION, reset=True)
    assert provider._drain_for_test()
    provider.prefetch("a real question about the loader")

    # The first-turn flag is per session, not per process.
    assert count_requests(fake_engram, "GET", "/context") == 2


def test_switch_resets_the_capture_high_water_mark(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The new session starts a new transcript; a high mark would make its early tool
    # results look "already seen".
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    long_turn = [tool_message("r" * 80) for _ in range(5)]
    provider.sync_turn("ask", "", session_id=SESSION, messages=long_turn)
    assert provider._drain_for_test()
    assert provider._synced_message_count == 5

    provider.on_session_switch(NEXT_SESSION, reset=True)
    assert provider._drain_for_test()
    assert provider._synced_message_count == 0
    fake_engram.reset()

    provider.sync_turn("ask", "", session_id=NEXT_SESSION, messages=[tool_message("n" * 80)])
    assert provider._drain_for_test()

    assert count_requests(fake_engram, "POST", "/observations/passive") == 1


def test_switch_resets_the_turn_counter_and_the_compaction_hash(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.on_turn_start(5, "a real question about the loader")
    provider.on_pre_compress(transcript())
    assert provider._drain_for_test()
    assert provider._turn_number == 5
    assert provider._last_pre_compress_hash != ""

    provider.on_session_switch(NEXT_SESSION, reset=True)
    assert provider._drain_for_test()

    assert provider._turn_number == 0
    assert provider._last_pre_compress_hash == ""


def test_switch_clears_the_recall_counters(fake_engram: FakeEngram, provider_factory: Any) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.prefetch("a real question about the loader")
    assert provider.recall_status() is not None

    provider.on_session_switch(NEXT_SESSION, reset=True)
    assert provider._drain_for_test()

    assert provider.recall_status() is None


def test_a_rewind_drops_turn_caches_without_ending_or_registering(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.prefetch("why does the loader skip my provider")
    assert provider.recall_status() is not None
    fake_engram.reset()

    provider.on_session_switch(SESSION, reset=False, rewound=True)
    assert provider._drain_for_test()

    assert provider.recall_status() is None
    assert end_requests(fake_engram) == []
    assert fake_engram.find("POST", "/sessions") is None
    assert provider._session_id == SESSION
    # The session and directory are unchanged, so the project stays detected.
    assert provider._project == PROJECT


def test_a_rewind_keeps_the_capture_mark_so_nothing_is_recaptured(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The transcript was truncated, but the observations captured from it are already
    # in Engram; zeroing the mark would re-post them.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.sync_turn("ask", "", session_id=SESSION, messages=[tool_message("r" * 80)] * 4)
    assert provider._drain_for_test()
    mark = provider._synced_message_count

    provider.on_session_switch(SESSION, reset=False, rewound=True)

    assert provider._synced_message_count == mark


def test_a_rewind_with_a_changed_session_id_is_a_real_switch(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_session_switch(NEXT_SESSION, reset=True, rewound=True)
    assert provider._drain_for_test()

    assert provider._session_id == NEXT_SESSION
    assert fake_engram.find("POST", "/sessions") is not None


def test_switch_to_the_same_id_does_not_end_it(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_session_switch(SESSION, reset=True)
    assert provider._drain_for_test()

    assert end_requests(fake_engram) == []


@pytest.mark.parametrize("context", ["subagent", "cron", "flush"])
def test_switch_writes_nothing_for_a_non_primary_context(
    fake_engram: FakeEngram, provider_factory: Any, context: str
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, agent_context=context)
    fake_engram.reset()

    provider.on_session_switch(NEXT_SESSION, reset=True)
    assert provider._drain_for_test()

    assert [r for r in fake_engram.requests if r.method == "POST"] == []


def test_the_ended_session_set_stays_bounded(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # A long-lived gateway process rotates sessions indefinitely; an unbounded set
    # would be a slow leak.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    for n in range(_ENDED_SESSION_MEMORY * 3):
        provider.on_session_switch(f"sess-{n}", reset=True)
        assert len(provider._ended_sessions) <= _ENDED_SESSION_MEMORY
    assert provider._drain_for_test(timeout=10.0)

    assert len(provider._ended_sessions) == _ENDED_SESSION_MEMORY


def test_switch_never_raises_on_garbage(fake_engram: FakeEngram, provider_factory: Any) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.on_session_switch(None)  # type: ignore[arg-type]
    provider.on_session_switch("", reset=True)
    provider.on_session_end(None)  # type: ignore[arg-type]

    assert provider._drain_for_test()


# -- 3. on_pre_compress: the archive ------------------------------------------


def test_on_pre_compress_archives_a_session_summary(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    returned = provider.on_pre_compress(transcript())

    requests = observation_requests(fake_engram)
    assert len(requests) == 1
    body = requests[0].json_body
    assert body["session_id"] == SESSION
    assert body["type"] == "session_summary"
    assert body["title"] == "Compaction recovery summary"
    assert body["project"] == PROJECT
    assert body["scope"] == "project"
    assert body["topic_key"] == "session/compaction-recovery"
    assert "question number 0" in body["content"]
    assert returned != ""


def test_on_pre_compress_archives_synchronously(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Unlike every other write in this provider: the return value has to tell the model
    # whether the summary is saved, and a queued write makes that unprovable.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_pre_compress(transcript())

    # Asserted BEFORE any drain.
    assert len(observation_requests(fake_engram)) == 1


def test_on_pre_compress_archives_once_for_the_same_window(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Compaction can fire twice over the same window; a duplicate session_summary is
    # noise in the user's store.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    messages = transcript()
    fake_engram.reset()

    first = provider.on_pre_compress(messages)
    second = provider.on_pre_compress(messages)

    assert len(observation_requests(fake_engram)) == 1
    # Both answers still tell the model the summary is saved.
    assert "already saved" in first
    assert "already saved" in second


def test_on_pre_compress_archives_again_for_a_different_window(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_pre_compress(transcript(2))
    provider.on_pre_compress(transcript(5))

    assert len(observation_requests(fake_engram)) == 2


def test_on_pre_compress_redacts_private_blocks(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = [
        {"role": "user", "content": "deploy with <private>hunter2</private> please"},
        {"role": "assistant", "content": "done"},
    ]

    provider.on_pre_compress(messages)

    content = observation_requests(fake_engram)[0].json_body["content"]
    assert "hunter2" not in content
    assert REDACTED in content


def test_on_pre_compress_caps_the_archived_content(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    huge = [{"role": "user", "content": "u" * 5000} for _ in range(200)] + [
        {"role": "assistant", "content": "a" * 50000}
    ]

    provider.on_pre_compress(huge)

    content = observation_requests(fake_engram)[0].json_body["content"]
    assert len(content) <= PRE_COMPRESS_MAX_CHARS + 3, len(content)


def test_on_pre_compress_digests_rather_than_copying_the_transcript(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()
    messages = transcript(40)

    provider.on_pre_compress(messages)

    content = observation_requests(fake_engram)[0].json_body["content"]
    assert "**Messages**: 80" in content
    # Only the last few user turns survive, not all forty.
    assert "question number 0 " not in content
    assert "question number 39" in content


def test_on_pre_compress_skips_an_empty_transcript(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    returned = provider.on_pre_compress([])

    assert observation_requests(fake_engram) == []
    # Nothing was archived, so the model is told to save it itself.
    assert "FIRST ACTION REQUIRED" in returned


# -- 4. on_pre_compress: the recovery text ------------------------------------


def test_on_pre_compress_includes_the_compaction_context(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, compaction_context="## Earlier\n- shipped the HTTP client")
    provider = provider_factory(fake_engram.base_url)

    returned = provider.on_pre_compress(transcript())

    assert "shipped the HTTP client" in returned
    assert fake_engram.find("GET", "/context/compaction") is not None


def test_on_pre_compress_tells_the_model_not_to_duplicate_a_saved_summary(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    returned = provider.on_pre_compress(transcript())

    assert "already saved" in returned
    assert "Do NOT call mem_session_summary" in returned


def test_on_pre_compress_tells_the_model_to_save_when_the_archive_was_rejected(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    fake_engram.route("POST", "/observations", (500, {"error": "boom"}))
    provider = provider_factory(fake_engram.base_url)

    returned = provider.on_pre_compress(transcript())

    # Definitely not written, so the instruction is unambiguous.
    assert "FIRST ACTION REQUIRED" in returned
    assert "mem_session_summary" in returned


def test_on_pre_compress_is_cautious_when_the_archive_write_timed_out(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A POST that timed out may still have been committed server-side, so telling the
    # model "save it yourself" would risk a duplicate.
    monkeypatch.setattr(client_module, "WRITE_TIMEOUT_S", 0.2)
    wire_engram(fake_engram)
    fake_engram.route("POST", "/observations", Delay(1.5, (201, {"id": 1, "status": "saved"})))
    provider = provider_factory(fake_engram.base_url)

    returned = provider.on_pre_compress(transcript())

    assert "could not be confirmed" in returned
    assert "Verify with mem_search first" in returned
    assert "FIRST ACTION REQUIRED" not in returned


def test_on_pre_compress_returns_the_instruction_alone_when_the_context_fails(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    fake_engram.route("GET", "/context/compaction", (500, {"error": "boom"}))
    provider = provider_factory(fake_engram.base_url)

    returned = provider.on_pre_compress(transcript())

    assert returned != ""
    assert "already saved" in returned
    assert "shipped the client" not in returned, "a stale context leaked in"


def test_on_pre_compress_handles_a_missing_compaction_context(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    fake_engram.route("GET", "/context/compaction", (404, {"error": "session not found"}))
    provider = provider_factory(fake_engram.base_url)

    assert provider.on_pre_compress(transcript()) != ""


def test_on_pre_compress_caps_the_returned_context(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, compaction_context="c" * 40000)
    provider = provider_factory(fake_engram.base_url)

    returned = provider.on_pre_compress(transcript())

    # The context section is everything before the blank line; the instruction block
    # that follows is a fixed, bounded addition. (Counting "c" across the whole string
    # would also count the ones in the instruction's own prose.)
    context_section = returned.split("\n\n")[0]
    assert set(context_section) == {"c", "."}, "the context section should be only the context"
    assert len(context_section) <= RECOVERY_MAX_CHARS + 3, len(context_section)
    assert len(returned) <= RECOVERY_MAX_CHARS + 1000, len(returned)


def test_on_pre_compress_emits_no_memory_context_tags(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    assert "<memory-context>" not in provider.on_pre_compress(transcript())


# -- 5. on_pre_compress: degraded paths ---------------------------------------


def test_on_pre_compress_returns_nothing_when_disabled(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, enabled=False)
    fake_engram.reset()

    assert provider.on_pre_compress(transcript()) == ""
    assert fake_engram.requests == []


@pytest.mark.parametrize("context", ["subagent", "cron", "flush"])
def test_on_pre_compress_returns_nothing_for_a_non_primary_context(
    fake_engram: FakeEngram, provider_factory: Any, context: str
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, agent_context=context)
    fake_engram.reset()

    assert provider.on_pre_compress(transcript()) == ""
    assert [r for r in fake_engram.requests if r.method == "POST"] == []


def test_on_pre_compress_returns_nothing_without_a_session(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, session_id="")
    fake_engram.reset()

    assert provider.on_pre_compress(transcript()) == ""
    assert fake_engram.requests == []


def test_on_pre_compress_returns_nothing_when_the_project_is_unknown(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, project="")
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    assert provider.on_pre_compress(transcript()) == ""
    assert observation_requests(fake_engram) == []


def test_on_pre_compress_never_raises_on_garbage(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    assert isinstance(provider.on_pre_compress(None), str)  # type: ignore[arg-type]
    assert isinstance(provider.on_pre_compress("not a list"), str)  # type: ignore[arg-type]
    assert isinstance(provider.on_pre_compress([None, 7, "x", {}]), str)  # type: ignore[list-item]


# -- 6. Engram down -----------------------------------------------------------


def test_every_lifecycle_hook_survives_an_absent_server(provider_factory: Any) -> None:
    provider = provider_factory(f"http://127.0.0.1:{unused_port()}")

    provider.on_session_end(transcript())
    provider.on_session_switch(NEXT_SESSION, reset=True)
    # The project can never be resolved, so there is nothing to contribute. This is the
    # documented degraded answer for a v1 provider: the host joins nothing.
    assert provider.on_pre_compress(transcript()) == ""

    assert provider._drain_for_test(timeout=10.0)
    provider.shutdown()
    assert provider._worker is None


def test_on_pre_compress_contributes_the_instruction_when_only_writes_fail(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Distinct from the case above: here the project IS known, so we can still tell the
    # model what to do about a summary that did not land.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.route("POST", "/observations", (503, {"error": "unavailable"}))
    fake_engram.route("GET", "/context/compaction", (503, {"error": "unavailable"}))

    returned = provider.on_pre_compress(transcript())

    assert returned != ""
    assert "FIRST ACTION REQUIRED" in returned


# -- 7. shutdown --------------------------------------------------------------


def test_shutdown_drains_a_queued_session_end(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The host calls on_session_end and then shutdown() at exit; the queued /end must
    # still land, without the test helper doing the draining.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_session_end(transcript())
    provider.shutdown()

    assert provider._worker is None
    assert [r.path for r in end_requests(fake_engram)] == [f"/sessions/{SESSION}/end"]


def test_shutdown_is_bounded_when_the_server_is_wedged(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A server that accepts the connection and never answers must not hold process
    # exit open: the join gives up after SHUTDOWN_DRAIN_TIMEOUT_S.
    monkeypatch.setattr(client_module, "WRITE_TIMEOUT_S", 10.0)
    wire_engram(fake_engram)
    fake_engram.route("POST", f"/sessions/{SESSION}/end", Delay(4.0, (200, {"id": SESSION})))
    provider = provider_factory(fake_engram.base_url)

    provider.on_session_end(transcript())
    started = time.monotonic()
    provider.shutdown()
    elapsed = time.monotonic() - started

    assert provider._worker is None
    assert elapsed < provider_module.SHUTDOWN_DRAIN_TIMEOUT_S + 0.5, elapsed


def test_the_shutdown_drain_stays_well_under_the_hosts_budget() -> None:
    # Hermes core gives its own executor a 5s drain before calling shutdown(); ours
    # comes on top of that while the user waits on process exit.
    assert 0 < provider_module.SHUTDOWN_DRAIN_TIMEOUT_S <= 2.5


# -- 8. invariants ------------------------------------------------------------


def test_pre_compress_checkpoint_api_version_is_still_one() -> None:
    # Version 2 is the opt-in fail-closed checkpoint contract: a v2 provider's
    # exception propagates and the host preserves the uncompressed transcript. We do
    # not implement that, so claiming v2 would make compaction fail closed on our
    # behalf without the guarantee behind it.
    assert RecallProvider.pre_compress_checkpoint_api_version == 1


def test_the_pre_compress_budget_is_bounded() -> None:
    # It runs on the caller thread at compaction time. Worst case is project detection
    # plus the archive write plus the recovery read, each individually bounded.
    assert provider_module.PRE_COMPRESS_BUDGET_S >= (
        client_module.READ_TIMEOUT_S + client_module.WRITE_TIMEOUT_S + client_module.READ_TIMEOUT_S
    )
    assert provider_module.PRE_COMPRESS_BUDGET_S <= 10.0


def test_reset_session_state_clears_every_per_session_field(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    """One place to notice a new per-session field that nobody wired into the reset."""
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.on_turn_start(4, "a real question about the loader")
    provider.prefetch("a real question about the loader")
    provider.queue_prefetch("another real question about the loader")
    provider.sync_turn("ask", "a" * 80, session_id=SESSION, messages=[tool_message("r" * 80)])
    provider.on_pre_compress(transcript())
    assert provider._drain_for_test()

    provider._reset_session_state()

    assert provider._project == ""
    assert provider._project_detected is False
    assert provider._context_fetched is False
    assert provider._registered_session_id == ""
    assert provider._capture_mark_pending is True
    assert provider._last_recall_count == 0
    assert provider._last_recall_injected is False
    assert provider._turn_number == 0
    assert provider._synced_message_count == 0
    assert provider._last_pre_compress_hash == ""


# -- 9. redaction runs before the digest's per-row cuts --------------------------

SECRET = "SECRETZ"


def private_straddling(cut: int) -> str:
    """Text whose ``<private>`` block opens just before ``cut`` and closes well after it."""
    return "a" * (cut - 20) + "<private>" + SECRET * 100 + "</private>" + " tail"


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"role": "user", "content": private_straddling(provider_module._DIGEST_USER_CHARS)},
            {"role": "assistant", "content": "answer with some detail"},
        ],
        [
            {"role": "user", "content": "question about the loader"},
            {
                "role": "assistant",
                "content": private_straddling(provider_module._DIGEST_ASSISTANT_CHARS),
            },
        ],
    ],
    ids=["user-turn-cut", "last-reply-cut"],
)
def test_on_pre_compress_does_not_leak_a_block_straddling_a_digest_cut(
    fake_engram: FakeEngram, provider_factory: Any, rows: list[dict[str, Any]]
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    provider.on_pre_compress(rows)

    content = observation_requests(fake_engram)[0].json_body["content"]
    assert "SECRET" not in content, "the straddling private block leaked"
    assert REDACTED in content


# -- 10. the compaction archive registers an unregistered session ---------------


def test_on_pre_compress_registers_and_retries_on_session_not_found(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    attempts: list[int] = []

    def observations(_request: RecordedRequest) -> tuple[int, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            return (404, {"error": "session not found"})
        return (201, {"id": 900, "status": "saved"})

    fake_engram.route("POST", "/observations", observations)
    provider = provider_factory(fake_engram.base_url)
    fake_engram.reset()

    notice = provider.on_pre_compress(transcript())

    assert len(attempts) == 2
    assert count_requests(fake_engram, "POST", "/sessions") == 1
    assert "already saved" in notice


# -- 11. the capture mark after a switch skips the inherited transcript ----------


def test_the_first_sync_after_a_switch_does_not_recapture_the_inherited_transcript(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Compression or /resume hands the new session a transcript that already holds
    # tool results captured under the old one. Only the turn that just ran is new.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "first question about the loader"},
        tool_message("o" * 80),
        tool_message("p" * 80),
        {"role": "assistant", "content": "first answer"},
    ]
    provider.sync_turn("ask", "", session_id=SESSION, messages=history)
    assert provider._drain_for_test()

    provider.on_session_switch(NEXT_SESSION, reset=False)
    assert provider._drain_for_test()
    fake_engram.reset()

    resumed = [
        *history,
        {"role": "user", "content": "second question about the loader"},
        tool_message("n" * 80),
    ]
    provider.sync_turn("ask", "", session_id=NEXT_SESSION, messages=resumed)
    assert provider._drain_for_test()

    contents = [r.json_body["content"] for r in fake_engram.requests if r.path.endswith("passive")]
    assert contents == ["n" * 80]
    assert provider._synced_message_count == len(resumed)


def test_the_first_sync_after_a_switch_captures_a_genuinely_new_transcript(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.sync_turn("ask", "", session_id=SESSION, messages=[tool_message("o" * 80)] * 3)
    assert provider._drain_for_test()

    provider.on_session_switch(NEXT_SESSION, reset=True)
    assert provider._drain_for_test()
    fake_engram.reset()

    fresh = [
        {"role": "user", "content": "a brand new question"},
        tool_message("n" * 80),
        tool_message("m" * 80),
    ]
    provider.sync_turn("ask", "", session_id=NEXT_SESSION, messages=fresh)
    assert provider._drain_for_test()

    contents = [r.json_body["content"] for r in fake_engram.requests if r.path.endswith("passive")]
    assert contents == ["n" * 80, "m" * 80]
