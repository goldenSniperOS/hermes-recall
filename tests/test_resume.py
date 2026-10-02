"""Effective Engram session ids: ``resume`` registration and ``<id>:resume:<N>``.

Engram v3 (6af3a4b) answers ``POST /sessions`` with ``"resume": true`` by selecting
the session to write into: the id itself while it is open (or new), otherwise the
lowest live ``<id>:resume:<N>`` continuation, or the next ordinal. Every write that
belongs to a Hermes session must then use that *effective* id. Engram 2.0.0 ignores
``resume`` and echoes the id, so there the effective id is the runtime id.

Nothing here touches the user's real Engram; background work is drained, never
slept on.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import pytest
from fake_engram import FakeEngram, RecordedRequest

from hermes_recall import provider as provider_module
from hermes_recall.config import RecallConfig
from hermes_recall.provider import RecallProvider

PROJECT = "hermes-recall"
SESSION = "sess-1"
NEXT_SESSION = "sess-2"
LONG = "x" * 80


def make_config(url: str) -> RecallConfig:
    return RecallConfig(
        url=url,
        project_override="",
        recall_limit=5,
        recall_max_chars=2500,
        enabled=True,
        auth_token="",
    )


class ResumingEngram:
    """Engram v3's resume selection over a FakeEngram (``store.continuationSessionTx``).

    ``ended`` maps every known session id to whether it has ended. ``resume=False``
    in the constructor emulates v2.2 (409 ``session_already_ended``); ``echo=True``
    emulates 2.0.0 (the id is echoed back whatever its state).
    """

    def __init__(self, fake: FakeEngram, *, resume: bool = True, echo: bool = False) -> None:
        self.fake = fake
        self.resume = resume
        self.echo = echo
        self.ended: dict[str, bool] = {}
        fake.route("GET", "/project/current", (200, {"project": PROJECT}))
        fake.route("POST", "/sessions", self.register)
        fake.route("POST", "/prompts", (201, {"id": 1, "status": "saved"}))
        fake.route("POST", "/observations", (201, {"id": 900, "status": "saved"}))
        fake.route(
            "POST", "/observations/passive", (200, {"extracted": 1, "saved": 1, "duplicates": 0})
        )
        fake.route("GET", "/context/compaction", (200, {"context": "## Earlier\n- x"}))
        fake.route(
            "POST",
            "/conflicts/judge",
            (200, {"relation": {"judgment_id": "rel-1", "relation": "related"}}),
        )
        fake.default_response = self.fallback  # type: ignore[assignment]

    def register(self, request: RecordedRequest) -> tuple[int, Any]:
        root = request.json_body["id"]
        if self.echo:
            self.ended.setdefault(root, False)
            return (201, {"id": root, "status": "created"})
        effective = root
        if self.ended.get(root):
            if not (self.resume and request.json_body.get("resume")):
                return (
                    409,
                    {
                        "error": "session has already ended",
                        "code": "session_already_ended",
                        "session_id": root,
                    },
                )
            prefix = f"{root}:resume:"
            ordinals = sorted(
                int(sid[len(prefix) :])
                for sid in self.ended
                if sid.startswith(prefix) and sid[len(prefix) :].isdigit()
            )
            live = [n for n in ordinals if not self.ended[f"{prefix}{n}"]]
            # The server's counter starts at 1, so the first continuation is :resume:2.
            effective = f"{prefix}{live[0] if live else max([1, *ordinals]) + 1}"
        self.ended.setdefault(effective, False)
        ack = {"id": effective, "status": "created"}
        if effective != root:
            ack["resumed_from"] = root
        return (201, ack)

    def fallback(self, request: RecordedRequest) -> tuple[int, Any]:
        path = unquote(request.path)
        if request.method == "POST" and path.startswith("/sessions/") and path.endswith("/end"):
            session_id = path[len("/sessions/") : -len("/end")]
            self.ended[session_id] = True
            return (200, {"id": session_id, "status": "completed"})
        return (404, {"error": "not found"})

    # -- what each write was attributed to ----------------------------------

    def write_sessions(self) -> list[tuple[str, str]]:
        """``(path, session_id)`` for every session-attributed request, in order."""
        rows: list[tuple[str, str]] = []
        for request in self.fake.requests:
            path = unquote(request.path)
            if request.method == "POST" and path in (
                "/prompts",
                "/observations",
                "/observations/passive",
                "/conflicts/judge",
            ):
                rows.append((path, request.json_body.get("session_id", "")))
            elif request.method == "GET" and path == "/context/compaction":
                rows.append((path, request.query_one("session_id") or ""))
            elif request.method == "POST" and path.startswith("/sessions/"):
                rows.append(("/end", path[len("/sessions/") : -len("/end")]))
        return rows

    def registrations(self) -> list[str]:
        return [
            r.json_body["id"]
            for r in self.fake.requests
            if r.method == "POST" and r.path == "/sessions"
        ]


@pytest.fixture
def make_provider(tmp_path: Path) -> Iterator[Any]:
    built: list[RecallProvider] = []

    def factory(url: str, *, session_id: str = SESSION) -> RecallProvider:
        provider = RecallProvider(config=make_config(url))
        built.append(provider)
        provider.initialize(session_id, hermes_home=str(tmp_path), platform="cli")
        assert provider._drain_for_test()
        return provider

    yield factory
    for provider in built:
        provider.shutdown()


def every_write(provider: RecallProvider) -> None:
    """Exercise each session-attributed write path once, and drain."""
    provider.on_turn_start(1, "why does the loader skip my provider")
    provider.sync_turn("ask", LONG, messages=[{"role": "tool", "content": LONG, "name": "read"}])
    provider.on_delegation("look at the loader please", LONG)
    assert provider._drain_for_test()
    for name, args in (
        ("mem_save", {"title": "T", "content": "C"}),
        ("mem_session_summary", {"content": "## Goal\nship"}),
        ("mem_judge", {"judgment_id": "rel-1", "relation": "related"}),
    ):
        assert "error" not in json.loads(provider.handle_tool_call(name, args)), name
    provider.on_pre_compress(
        [{"role": "user", "content": "a question"}, {"role": "assistant", "content": "an answer"}]
    )


EVERY_WRITE_PATH = [
    "/prompts",
    "/observations/passive",
    "/observations/passive",
    "/observations/passive",
    "/observations",
    "/observations",
    "/conflicts/judge",
    "/observations",
    "/context/compaction",
]


# -- 1. the acknowledged id is the effective session -------------------------------


def test_a_continuation_ack_is_used_by_every_session_write(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    engram = ResumingEngram(fake_engram)
    engram.ended[SESSION] = True  # Hermes resumed a session that already ended.
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    every_write(provider)
    provider.on_session_end([])
    assert provider._drain_for_test()

    continuation = f"{SESSION}:resume:2"
    rows = engram.write_sessions()
    assert [path for path, _ in rows] == [*EVERY_WRITE_PATH, "/end"]
    assert {sid for _, sid in rows} == {continuation}
    # Registered once, at initialize: the writes never re-registered.
    assert engram.registrations() == []
    assert engram.ended == {SESSION: True, continuation: True}


def test_engram_2_echo_keeps_the_runtime_id_and_registers_once(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    engram = ResumingEngram(fake_engram, echo=True)
    provider = make_provider(fake_engram.base_url)
    assert engram.registrations() == [SESSION]
    fake_engram.reset()

    every_write(provider)
    provider.on_session_end([])
    assert provider._drain_for_test()

    rows = engram.write_sessions()
    assert [path for path, _ in rows] == [*EVERY_WRITE_PATH, "/end"]
    assert {sid for _, sid in rows} == {SESSION}
    assert engram.registrations() == []


@pytest.mark.parametrize(
    "ack",
    [
        {"id": NEXT_SESSION, "status": "created"},
        {"id": f"{SESSION}x", "status": "created"},
        {"id": f"{SESSION}:resume:", "status": "created"},
        {"id": f"{SESSION}:resume:2x", "status": "created"},
        {"id": f"{SESSION}:resume:2:resume:3", "status": "created"},
        {"id": f"{SESSION}:resume:\u00b2", "status": "created"},
        {"id": f"{SESSION}:resume:2", "status": "ok"},
        {"id": f"{SESSION}:resume:2"},
        {"id": 7, "status": "created"},
        {"status": "created"},
        {"id": SESSION, "status": None},
    ],
    ids=[
        "other-session",
        "prefix-only",
        "empty-ordinal",
        "non-numeric-ordinal",
        "nested",
        "unicode-digit",
        "wrong-status",
        "no-status",
        "non-string-id",
        "no-id",
        "null-status",
    ],
)
def test_an_invalid_ack_is_not_a_registration_and_its_id_is_never_written(
    fake_engram: FakeEngram, make_provider: Any, ack: dict[str, Any]
) -> None:
    engram = ResumingEngram(fake_engram)
    fake_engram.route("POST", "/sessions", (201, ack))
    provider = make_provider(fake_engram.base_url)

    assert provider._registered_session_id == ""
    assert provider._effective_session_id(SESSION) == SESSION
    fake_engram.reset()

    provider.on_turn_start(1, "why does the loader skip my provider")
    provider.on_delegation("look at the loader please", LONG)
    assert provider._drain_for_test()

    # Not registered, so each job tries again — exactly once per job.
    assert len(engram.registrations()) == 2
    assert {sid for _, sid in engram.write_sessions()} == {SESSION}


# -- 2. end, then resume the same Hermes session ------------------------------------


def test_resuming_an_ended_session_writes_into_a_continuation(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    engram = ResumingEngram(fake_engram)
    provider = make_provider(fake_engram.base_url)
    provider.on_turn_start(1, "first question about the loader")
    assert provider._drain_for_test()
    # The host's rotation order for /new: end, then switch with reset.
    provider.on_session_end([])
    provider.on_session_switch(NEXT_SESSION, reset=True)
    assert provider._drain_for_test()
    # /resume back into the first session.
    provider.on_session_switch(SESSION, reset=False)
    provider.on_turn_start(2, "second question about the loader")
    assert provider._drain_for_test()

    continuation = f"{SESSION}:resume:2"
    assert engram.registrations() == [SESSION, NEXT_SESSION, SESSION]
    assert [r for r in engram.write_sessions() if r[0] == "/prompts"] == [
        ("/prompts", SESSION),
        ("/prompts", continuation),
    ]

    # Ending it again closes the continuation, exactly once, even though the root id
    # is still in the recently-ended set.
    provider.on_session_end([])
    provider.on_session_switch("sess-3", reset=True)
    assert provider._drain_for_test()

    ends = [sid for path, sid in engram.write_sessions() if path == "/end"]
    assert ends == [SESSION, continuation]
    assert engram.ended[continuation] is True


def test_a_reset_away_from_a_resumed_session_ends_its_continuation(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    # No on_session_end this time: only the switch's own end can close the
    # continuation, and the root id is still in the recently-ended set.
    engram = ResumingEngram(fake_engram)
    provider = make_provider(fake_engram.base_url)
    provider.on_session_end([])
    provider.on_session_switch(NEXT_SESSION, reset=True)
    provider.on_session_switch(SESSION, reset=False)
    assert provider._drain_for_test()

    provider.on_session_switch("sess-3", reset=True)
    assert provider._drain_for_test()

    ends = [sid for path, sid in engram.write_sessions() if path == "/end"]
    assert ends == [SESSION, f"{SESSION}:resume:2"]


def test_a_write_after_the_end_of_the_same_session_lands_in_a_continuation(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    engram = ResumingEngram(fake_engram)
    provider = make_provider(fake_engram.base_url)
    provider.on_session_end([])
    assert provider._drain_for_test()
    fake_engram.reset()

    provider.on_turn_start(2, "a question asked after the end")
    assert provider._drain_for_test()

    assert engram.registrations() == [SESSION]
    assert engram.write_sessions() == [("/prompts", f"{SESSION}:resume:2")]


def test_registering_the_same_ended_session_again_reuses_the_live_continuation(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    # Deterministic and stateless server-side: a re-registration (here a switch away
    # and back without an end) gets the same continuation, never a new one.
    engram = ResumingEngram(fake_engram)
    engram.ended[SESSION] = True
    provider = make_provider(fake_engram.base_url)
    provider.on_session_switch(NEXT_SESSION, reset=False)
    provider.on_session_switch(SESSION, reset=False)
    provider.on_turn_start(1, "why does the loader skip my provider")
    assert provider._drain_for_test()

    assert engram.write_sessions() == [("/prompts", f"{SESSION}:resume:2")]
    assert f"{SESSION}:resume:3" not in engram.ended


def test_v2_2_refusing_an_ended_session_falls_back_to_the_runtime_id(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    # v2.2 ignores `resume` and answers 409 session_already_ended: not registered, and
    # the write goes where it always went.
    engram = ResumingEngram(fake_engram, resume=False)
    engram.ended[SESSION] = True
    provider = make_provider(fake_engram.base_url)

    provider.on_turn_start(1, "why does the loader skip my provider")
    assert provider._drain_for_test()

    assert provider._registered_session_id == ""
    assert engram.write_sessions() == [("/prompts", SESSION)]


# -- 3. jobs never cross sessions ---------------------------------------------------


def test_a_job_queued_for_one_session_never_writes_into_the_next_ones_effective_id(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    engram = ResumingEngram(fake_engram)
    engram.ended[SESSION] = True
    engram.ended[NEXT_SESSION] = True
    provider = make_provider(fake_engram.base_url)

    gate = threading.Event()
    provider._submit("hold", lambda: gate.wait(5))
    provider.on_turn_start(1, "question asked in the first session")
    provider.on_session_switch(NEXT_SESSION, reset=False)
    provider.on_turn_start(1, "question asked in the second session")
    gate.set()
    assert provider._drain_for_test()

    assert [r for r in engram.write_sessions() if r[0] == "/prompts"] == [
        ("/prompts", f"{SESSION}:resume:2"),
        ("/prompts", f"{NEXT_SESSION}:resume:2"),
    ]


def test_a_session_not_found_retry_writes_into_the_freshly_acknowledged_id(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    # Registration failed at startup, so the first write targets the runtime id; the
    # retry after "session not found" must use whatever the registration selected.
    engram = ResumingEngram(fake_engram)
    engram.ended[SESSION] = True
    fake_engram.route("POST", "/sessions", (500, {"error": "boom"}))
    provider = make_provider(fake_engram.base_url)
    fake_engram.route("POST", "/sessions", engram.register)
    fake_engram.reset()

    def prompts(request: RecordedRequest) -> tuple[int, Any]:
        if request.json_body["session_id"] == SESSION:
            return (404, {"error": "session not found"})
        return (201, {"id": 1, "status": "saved"})

    fake_engram.route("POST", "/prompts", prompts)
    provider._registered_session_id = SESSION  # skip the pre-registration
    provider.on_turn_start(1, "why does the loader skip my provider")
    assert provider._drain_for_test()
    attempts = [sid for path, sid in engram.write_sessions() if path == "/prompts"]

    assert attempts == [SESSION, f"{SESSION}:resume:2"]
    assert engram.registrations() == [SESSION]


# -- 4. bookkeeping ---------------------------------------------------------------


def test_the_effective_session_map_stays_bounded(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    ResumingEngram(fake_engram, echo=True)
    provider = make_provider(fake_engram.base_url)
    for n in range(provider_module._EFFECTIVE_SESSION_MEMORY + 10):
        # Drained each time: the register job registers whichever session is current.
        provider.on_session_switch(f"s-{n}", reset=False)
        assert provider._drain_for_test()

    assert len(provider._effective_sessions) == provider_module._EFFECTIVE_SESSION_MEMORY
    last = f"s-{provider_module._EFFECTIVE_SESSION_MEMORY + 9}"
    assert provider._effective_sessions[last] == last


def test_the_session_end_path_quotes_a_continuation_id(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    engram = ResumingEngram(fake_engram)
    engram.ended[SESSION] = True
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    provider.on_session_end([])
    assert provider._drain_for_test()

    continuation = f"{SESSION}:resume:2"
    assert fake_engram.find("POST", f"/sessions/{quote(continuation, safe='')}/end")
