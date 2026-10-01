"""Behaviour of ``hermes_recall.provider.RecallProvider`` against a fake Engram server.

Nothing here touches the user's real Engram on port 7437: every test points the
provider at a :class:`~fake_engram.FakeEngram` on an ephemeral port, or at a port
with nothing listening on it.

Background work is always *drained* (``_drain_for_test``) rather than slept on, so
the assertions are deterministic.
"""

from __future__ import annotations

import os
import re
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from fake_engram import Delay, FakeEngram

from hermes_recall import client as client_module
from hermes_recall import provider as provider_module
from hermes_recall._hermes import RecallStatus
from hermes_recall.config import RecallConfig
from hermes_recall.provider import (
    PREFETCH_BUDGET_S,
    QUERY_MAX_CHARS,
    SYSTEM_PROMPT_BLOCK,
    RecallProvider,
)

# -- helpers ------------------------------------------------------------------

PROJECT = "hermes-recall"


def make_config(url: str, **over: Any) -> RecallConfig:
    """A fully specified config, so no test depends on the ambient environment."""
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


def hit(identifier: int, *, content: str = "**What**: a thing") -> dict[str, Any]:
    """A search hit shaped like a live Engram answer."""
    return {
        "id": identifier,
        "type": "decision",
        "title": f"Memory {identifier}",
        "content": content,
        "project": PROJECT,
    }


def wire_engram(
    fake: FakeEngram,
    *,
    project: str = PROJECT,
    hits: list[dict[str, Any]] | None = None,
    context: str = "## Recent\n- did a thing",
) -> None:
    """Route the four endpoints T3 touches."""
    fake.route("GET", "/project/current", (200, {"project": project, "project_source": "cwd"}))
    fake.route("GET", "/search", (200, hits if hits is not None else [hit(761)]))
    fake.route("GET", "/context", (200, {"context": context}))
    fake.route("POST", "/sessions", (201, {"id": "sess-1", "status": "created"}))


def count_requests(fake: FakeEngram, method: str, path: str) -> int:
    """How many times ``(method, path)`` was recorded."""
    return sum(1 for r in fake.requests if r.method == method.upper() and r.path == path)


def unused_port() -> int:
    """Bind port 0, read the port, release it — nothing will be listening there."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def provider_factory() -> Iterator[Any]:
    """Build providers and shut their workers down on teardown."""
    built: list[RecallProvider] = []

    def build(url: str, **config_over: Any) -> RecallProvider:
        provider = RecallProvider(config=make_config(url, **config_over))
        built.append(provider)
        return provider

    try:
        yield build
    finally:
        for provider in built:
            provider.shutdown()


# -- 1. identity and availability ---------------------------------------------


def test_name_is_recall() -> None:
    # The host resolves the provider by install-directory / entry-point name.
    assert RecallProvider(config=make_config("http://127.0.0.1:1")).name == "recall"


def test_is_available_by_default() -> None:
    assert RecallProvider(config=make_config("http://127.0.0.1:1")).is_available() is True


def test_is_available_is_false_when_disabled() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1", enabled=False))
    assert provider.is_available() is False


def test_is_available_returns_false_instead_of_raising_on_a_broken_config() -> None:
    class Boom:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError(f"config is broken: {name}")

    # The client is injected so construction never reads the broken config.
    provider = RecallProvider(config=Boom(), client=object())  # type: ignore[arg-type]

    assert provider.is_available() is False


def test_is_available_makes_no_request(fake_engram: FakeEngram) -> None:
    wire_engram(fake_engram)

    assert RecallProvider(config=make_config(fake_engram.base_url)).is_available() is True
    assert fake_engram.requests == []


def test_unavailable_reason_names_the_config_key_when_disabled() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1", enabled=False))

    reason = provider.unavailable_reason()
    assert "memory.recall.enabled" in reason


def test_unavailable_reason_points_at_engram_serve_otherwise() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:7437"))

    reason = provider.unavailable_reason()
    assert "engram serve" in reason
    assert "http://127.0.0.1:7437" in reason


def test_unavailable_reason_never_raises_on_a_broken_config() -> None:
    class Boom:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("nope")

    provider = RecallProvider(config=Boom(), client=object())  # type: ignore[arg-type]
    assert provider.unavailable_reason() != ""


def test_construction_does_no_io(fake_engram: FakeEngram) -> None:
    # discover_memory_providers instantiates the class just to ask is_available().
    wire_engram(fake_engram)

    RecallProvider(config=make_config(fake_engram.base_url))

    assert fake_engram.requests == []


# -- 2. initialize ------------------------------------------------------------


def test_initialize_registers_the_session(fake_engram: FakeEngram, provider_factory: Any) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.initialize(session_id="sess-1", hermes_home="/tmp/hermes", platform="cli")
    assert provider._drain_for_test()

    request = fake_engram.find("POST", "/sessions")
    assert request is not None
    assert request.json_body["id"] == "sess-1"
    assert request.json_body["project"] == PROJECT


def test_initialize_stores_the_runtime_context(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.initialize(
        session_id="sess-1",
        hermes_home="/tmp/hermes",
        platform="telegram",
        agent_context="primary",
    )
    assert provider._drain_for_test()

    assert provider._session_id == "sess-1"
    assert provider._hermes_home == "/tmp/hermes"
    assert provider._platform == "telegram"
    assert provider._agent_context == "primary"
    assert provider._writes_enabled is True


@pytest.mark.parametrize("context", ["subagent", "cron", "flush"])
def test_initialize_performs_no_writes_for_a_non_primary_context(
    fake_engram: FakeEngram, provider_factory: Any, context: str
) -> None:
    # A cron system prompt or a subagent transcript would pollute the user's
    # project memory, so every write path is off for non-primary contexts.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.initialize(
        session_id="sess-1", hermes_home="/tmp/hermes", platform="cli", agent_context=context
    )
    assert provider._drain_for_test()

    assert provider._writes_enabled is False
    posts = [r for r in fake_engram.requests if r.method == "POST"]
    assert posts == [], posts


def test_initialize_is_a_no_op_when_disabled(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, enabled=False)

    provider.initialize(session_id="sess-1", hermes_home="/tmp/h", platform="cli")
    assert provider._drain_for_test()

    assert fake_engram.requests == []


def test_initialize_does_not_raise_with_no_server_listening(provider_factory: Any) -> None:
    provider = provider_factory(f"http://127.0.0.1:{unused_port()}")

    provider.initialize(session_id="sess-1", hermes_home="/tmp/h", platform="cli")

    assert provider._drain_for_test()
    assert provider.is_available() is True


def test_initialize_tolerates_a_blank_session_id(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.initialize(session_id="", hermes_home="/tmp/h", platform="cli")
    assert provider._drain_for_test()

    # The server requires an id, so there is nothing to register.
    assert fake_engram.find("POST", "/sessions") is None


# -- 3. project detection -----------------------------------------------------


def test_project_detection_sends_the_cwd(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.prefetch("why does recall skip my provider")

    request = fake_engram.find("GET", "/project/current")
    assert request is not None
    assert request.query_one("cwd") == os.getcwd()


def test_terminal_cwd_wins_over_getcwd(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERMINAL_CWD", "/Users/dev/somewhere-else")
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.prefetch("why does recall skip my provider")

    request = fake_engram.find("GET", "/project/current")
    assert request is not None
    assert request.query_one("cwd") == "/Users/dev/somewhere-else"


def test_project_override_short_circuits_detection(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, project_override="explicit-project")

    provider.prefetch("why does recall skip my provider")

    assert fake_engram.find("GET", "/project/current") is None
    search = fake_engram.find("GET", "/search")
    assert search is not None
    assert search.query_one("project") == "explicit-project"


def test_a_blank_project_is_unknown_and_does_not_crash(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Engram answers an ambiguous project with a 200 and a blank `project`.
    fake_engram.route(
        "GET",
        "/project/current",
        (200, {"project": "", "available_projects": ["a", "b"], "error_hint": "ambiguous"}),
    )
    provider = provider_factory(fake_engram.base_url)

    assert provider.prefetch("why does recall skip my provider") == ""
    assert provider.recall_status() is None
    assert fake_engram.find("GET", "/search") is None


def test_project_is_detected_once_per_session(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.prefetch("first real question about the loader")
    provider.prefetch("second real question about the loader")
    provider.prefetch("third real question about the loader")

    assert count_requests(fake_engram, "GET", "/project/current") == 1


def test_reset_project_cache_forces_re_detection(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.prefetch("first real question about the loader")
    provider._reset_project_cache()
    provider.prefetch("second real question about the loader")

    assert count_requests(fake_engram, "GET", "/project/current") == 2


# -- 4. system prompt block ---------------------------------------------------


def test_system_prompt_block_fits_the_budget() -> None:
    # Hard ceiling: this block ships on every API call, so it is budgeted, not
    # "kept short". Currently 1235 chars.
    assert 0 < len(SYSTEM_PROMPT_BLOCK) <= 1500, len(SYSTEM_PROMPT_BLOCK)


def test_system_prompt_block_is_byte_identical_across_calls() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1"))

    first = provider.system_prompt_block()
    second = provider.system_prompt_block()

    assert first == second
    assert first is SYSTEM_PROMPT_BLOCK


def test_system_prompt_block_carries_nothing_dynamic() -> None:
    # Hermes caches the assembled prompt prefix; a project name, an id, a count or
    # a timestamp in here would bust that cache on every turn it changed. No digit
    # at all is the strongest form of "nothing that looks like an id" — `#` itself
    # cannot be banned, since the block is Markdown and uses it for headings.
    digits = [ch for ch in SYSTEM_PROMPT_BLOCK if ch.isdigit()]
    assert digits == [], digits
    assert re.search(r"#\d", SYSTEM_PROMPT_BLOCK) is None
    assert PROJECT not in SYSTEM_PROMPT_BLOCK


def test_system_prompt_block_covers_the_contract() -> None:
    block = SYSTEM_PROMPT_BLOCK

    assert "mem_save" in block
    assert "mem_session_summary" in block
    assert "mem_search" in block
    assert "mem_context" in block
    # Recall is automatic; the model must not search just to orient itself.
    assert "automatically" in block
    for trigger in ("bug fix", "decision", "discovery", "configuration", "pattern", "preference"):
        assert trigger in block, trigger
    for section in ("**What**", "**Why**", "**Where**", "**Learned**"):
        assert section in block, section
    for field_name in ("**title**", "**type**", "**scope**"):
        assert field_name in block, field_name


def test_system_prompt_block_emits_no_memory_context_tags() -> None:
    assert "<memory-context>" not in SYSTEM_PROMPT_BLOCK


def test_system_prompt_block_is_empty_when_disabled() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1", enabled=False))
    assert provider.system_prompt_block() == ""


# -- 5. prefetch --------------------------------------------------------------


def test_prefetch_builds_the_search_request(fake_engram: FakeEngram, provider_factory: Any) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, recall_limit=7)

    text = provider.prefetch("why does the loader skip my provider")

    assert "#761" in text
    request = fake_engram.find("GET", "/search")
    assert request is not None
    assert request.query_one("q") == "why does the loader skip my provider"
    assert request.query_one("project") == PROJECT
    assert request.query_one("limit") == "7"
    assert request.query_one("match_mode") == "any"


def test_prefetch_truncates_a_long_query(fake_engram: FakeEngram, provider_factory: Any) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    long_query = "  " + ("the loader skips my provider because " * 40) + "  "

    provider.prefetch(long_query)

    request = fake_engram.find("GET", "/search")
    assert request is not None
    recorded = request.query_one("q")
    assert recorded is not None
    assert len(recorded) == QUERY_MAX_CHARS == 200
    assert recorded == long_query.strip()[:QUERY_MAX_CHARS]


@pytest.mark.parametrize("trivial", ["", "   ", "ok", "hi!", "/reset", "thanks :)", "done???"])
def test_prefetch_on_a_trivial_prompt_makes_no_request(
    fake_engram: FakeEngram, provider_factory: Any, trivial: str
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    assert provider.prefetch(trivial) == ""
    assert fake_engram.requests == []


def test_prefetch_returns_nothing_when_disabled(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, enabled=False)

    assert provider.prefetch("a real question about the loader") == ""
    assert fake_engram.requests == []


def test_prefetch_fetches_the_project_context_once_per_session(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.prefetch("first real question about the loader")
    provider.prefetch("second real question about the loader")
    provider.prefetch("third real question about the loader")

    assert count_requests(fake_engram, "GET", "/context") == 1
    assert count_requests(fake_engram, "GET", "/search") == 3


def test_prefetch_puts_the_context_in_the_first_turns_block(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, context="## Recent\n- shipped the client")
    provider = provider_factory(fake_engram.base_url)

    first = provider.prefetch("first real question about the loader")
    second = provider.prefetch("second real question about the loader")

    assert "shipped the client" in first
    assert "shipped the client" not in second
    assert "#761" in second


# -- 5b. prefetch wall-clock budget -------------------------------------------


def request_order(fake: FakeEngram) -> list[tuple[str, str]]:
    """Every recorded request as ``(method, path)``, in arrival order."""
    return [(r.method, r.path) for r in fake.requests]


def test_prefetch_budget_leaves_headroom_under_the_host_join() -> None:
    # Hermes joins an external prefetch for 8s and then skips the provider for the
    # REST of the session, so the whole synchronous path must finish well inside it.
    assert PREFETCH_BUDGET_S < 8.0
    # And the budget must be able to absorb both mandatory calls (project detection
    # plus the search), or the budget itself would be what stops recall working.
    assert PREFETCH_BUDGET_S >= 2 * client_module.READ_TIMEOUT_S


def test_prefetch_searches_before_fetching_the_project_context(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The search is the turn's actual recall value; /context is a one-off
    # nice-to-have. Under a budget the optional one must be the one that is dropped,
    # which only works if it runs last.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.prefetch("a real question about the loader")

    assert request_order(fake_engram) == [
        ("GET", "/project/current"),
        ("GET", "/search"),
        ("GET", "/context"),
    ]


def test_prefetch_skips_the_search_when_project_detection_ate_the_budget(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_module, "PREFETCH_BUDGET_S", 0.05)
    wire_engram(fake_engram)
    fake_engram.route("GET", "/project/current", Delay(0.3, (200, {"project": PROJECT})))
    provider = provider_factory(fake_engram.base_url)

    assert provider.prefetch("a real question about the loader") == ""
    assert provider.recall_status() is None
    # Asserted from the request log, not from wall-clock timing.
    assert request_order(fake_engram) == [("GET", "/project/current")]


def test_prefetch_returns_what_fit_when_the_search_ate_the_budget(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_module, "PREFETCH_BUDGET_S", 0.05)
    wire_engram(fake_engram, context="## Recent\n- shipped the client")
    fake_engram.route("GET", "/search", Delay(0.3, (200, [hit(761)])))
    provider = provider_factory(fake_engram.base_url)

    text = provider.prefetch("a real question about the loader")

    # Built from what fit: the hits landed, the optional context never started.
    assert "#761" in text
    assert "shipped the client" not in text
    assert request_order(fake_engram) == [("GET", "/project/current"), ("GET", "/search")]
    status = provider.recall_status()
    assert status is not None and status.count == 1


def test_a_context_fetch_skipped_for_budget_stays_pending_for_a_later_turn(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fetch that was never ATTEMPTED is not evidence the server cannot answer, so
    # unlike a failed fetch it must stay pending.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    monkeypatch.setattr(provider_module, "PREFETCH_BUDGET_S", 0.05)
    assert provider.prefetch("first real question about the loader") != ""
    assert count_requests(fake_engram, "GET", "/context") == 0

    monkeypatch.setattr(provider_module, "PREFETCH_BUDGET_S", 4.0)
    second = provider.prefetch("second real question about the loader")

    assert count_requests(fake_engram, "GET", "/context") == 1
    assert "did a thing" in second


def test_a_context_fetch_that_failed_is_not_retried(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The other half of the distinction: this one WAS attempted, so a later turn
    # must not keep paying for it.
    wire_engram(fake_engram)
    fake_engram.route("GET", "/context", (500, {"error": "boom"}))
    provider = provider_factory(fake_engram.base_url)

    provider.prefetch("first real question about the loader")
    provider.prefetch("second real question about the loader")
    provider.prefetch("third real question about the loader")

    assert count_requests(fake_engram, "GET", "/context") == 1
    assert count_requests(fake_engram, "GET", "/search") == 3


def test_queue_prefetch_runs_without_a_deadline(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The background job is off the turn's critical path, so a spent turn budget
    # must not stop it from fetching the project context.
    monkeypatch.setattr(provider_module, "PREFETCH_BUDGET_S", 0.0)
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.queue_prefetch("why does the loader skip my provider")
    assert provider._drain_for_test()

    assert count_requests(fake_engram, "GET", "/search") == 1
    assert count_requests(fake_engram, "GET", "/context") == 1


def test_a_zero_budget_turn_recalls_nothing_and_reports_nothing(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    # Detect the project first, so the only call left for the budget to stop is the
    # search itself.
    assert provider.prefetch("a first real question about the loader") != ""
    fake_engram.reset()

    monkeypatch.setattr(provider_module, "PREFETCH_BUDGET_S", -1.0)
    assert provider.prefetch("a second real question about the loader") == ""

    assert provider.recall_status() is None
    assert fake_engram.requests == []


def test_a_spent_budget_still_serves_a_cached_recall(
    fake_engram: FakeEngram, provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The budget bounds network work. A queued result is already in hand, so
    # withholding it would cost the user recall for no benefit at all.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.queue_prefetch("why does the loader skip my provider")
    assert provider._drain_for_test()
    fake_engram.reset()

    monkeypatch.setattr(provider_module, "PREFETCH_BUDGET_S", -1.0)
    text = provider.prefetch("why does the loader skip my provider")

    assert "#761" in text
    status = provider.recall_status()
    assert status is not None and status.count == 1
    assert fake_engram.requests == []


def test_prefetch_never_exceeds_the_char_budget(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    hits = [hit(n, content="z" * 2000) for n in range(1, 16)]
    wire_engram(fake_engram, hits=hits, context="c" * 4000)
    provider = provider_factory(fake_engram.base_url, recall_max_chars=300, recall_limit=15)

    text = provider.prefetch("a real question about the loader")

    assert len(text) <= 300, len(text)


def test_prefetch_emits_no_memory_context_tags(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    text = provider.prefetch("a real question about the loader")

    assert "<memory-context>" not in text


def test_prefetch_returns_nothing_when_there_are_no_hits(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, hits=[], context="")
    provider = provider_factory(fake_engram.base_url)

    assert provider.prefetch("a real question about the loader") == ""


@pytest.mark.parametrize(
    "route",
    [(500, {"error": "boom"}), (200, {"not": "a list"}), (400, {"error": "bad q"})],
    ids=["500", "dict-body", "400"],
)
def test_prefetch_degrades_to_nothing_on_a_broken_search(
    fake_engram: FakeEngram, provider_factory: Any, route: Any
) -> None:
    wire_engram(fake_engram, context="")
    fake_engram.route("GET", "/search", route)
    provider = provider_factory(fake_engram.base_url)

    assert provider.prefetch("a real question about the loader") == ""
    assert provider.recall_status() is None


def test_prefetch_accepts_an_explicit_session_id(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # prefetch_all is called positionally, so session_id arrives as "" in the CLI
    # path; a gateway does pass one, and it must not change the answer's shape.
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.initialize(session_id="sess-1", hermes_home="/tmp/h", platform="cli")
    assert provider._drain_for_test()

    assert "#761" in provider.prefetch("a real question", session_id="gateway-session")
    assert provider._resolve_session_id("") == "sess-1"
    assert provider._resolve_session_id("  gw  ") == "gw"


# -- 6. Engram down -----------------------------------------------------------


def test_prefetch_is_silent_with_no_server_listening(provider_factory: Any) -> None:
    provider = provider_factory(f"http://127.0.0.1:{unused_port()}")

    assert provider.prefetch("a real question about the loader") == ""
    assert provider.recall_status() is None
    # Availability is config-only, never a health check.
    assert provider.is_available() is True


# -- 7. recall_status ---------------------------------------------------------


def test_recall_status_is_none_before_any_prefetch(provider_factory: Any) -> None:
    assert provider_factory("http://127.0.0.1:1").recall_status() is None


def test_recall_status_after_a_prefetch_with_hits(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, hits=[hit(1), hit(2), hit(3)], context="")
    provider = provider_factory(fake_engram.base_url)

    provider.prefetch("a real question about the loader")
    status = provider.recall_status()

    assert isinstance(status, RecallStatus)
    assert status.count == 3
    assert status.provider_label == f"Recall · {PROJECT}"
    assert status.glyph == "🧠"


def test_recall_status_label_omits_an_unknown_project(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # Project override keeps detection out of the way; here the project is blank
    # only because the recall came from the cache before detection memoised it.
    wire_engram(fake_engram, hits=[hit(1)], context="")
    provider = provider_factory(fake_engram.base_url, project_override="p")
    provider.prefetch("a real question about the loader")
    provider._project = ""

    status = provider.recall_status()
    assert status is not None
    assert status.provider_label == "Recall"


def test_recall_status_is_none_after_a_prefetch_with_zero_hits(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, hits=[], context="")
    provider = provider_factory(fake_engram.base_url)

    assert provider.prefetch("a real question about the loader") == ""
    assert provider.recall_status() is None


def test_recall_status_does_not_report_a_stale_count_after_a_trivial_turn(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # The important one: describe_recall only runs when prefetch returned text, but
    # recall_status must reflect ONLY the last prefetch, never a prior count.
    wire_engram(fake_engram, hits=[hit(1), hit(2)], context="")
    provider = provider_factory(fake_engram.base_url)

    assert provider.prefetch("a real question about the loader") != ""
    first = provider.recall_status()
    assert first is not None and first.count == 2

    assert provider.prefetch("ok") == ""
    assert provider.recall_status() is None


def test_recall_status_does_not_report_a_stale_count_after_a_failed_turn(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram, hits=[hit(1), hit(2)], context="")
    provider = provider_factory(fake_engram.base_url)
    assert provider.prefetch("a real question about the loader") != ""

    fake_engram.route("GET", "/search", (500, {"error": "boom"}))
    assert provider.prefetch("another real question about the loader") == ""
    assert provider.recall_status() is None


def test_recall_status_reports_zero_for_a_context_only_block(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    # count <= 0 means "content injected, no discrete count"; the host renders it
    # as "recalled relevant memory" rather than "recalled 0 memories".
    wire_engram(fake_engram, hits=[], context="## Recent\n- shipped the client")
    provider = provider_factory(fake_engram.base_url)

    assert provider.prefetch("a real question about the loader") != ""
    status = provider.recall_status()
    assert status is not None
    assert status.count == 0


# -- 8. queue_prefetch --------------------------------------------------------


def test_queue_prefetch_serves_the_next_prefetch_from_cache(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.queue_prefetch("why does the loader skip my provider")
    assert provider._drain_for_test()
    assert count_requests(fake_engram, "GET", "/search") == 1

    text = provider.prefetch("why does the loader skip my provider")

    assert "#761" in text
    assert count_requests(fake_engram, "GET", "/search") == 1, "cached result was not used"
    status = provider.recall_status()
    assert status is not None and status.count == 1


def test_queue_prefetch_cache_is_never_served_for_a_different_query(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.queue_prefetch("why does the loader skip my provider")
    assert provider._drain_for_test()

    provider.prefetch("something else entirely about the client")

    assert count_requests(fake_engram, "GET", "/search") == 2


def test_queue_prefetch_normalises_the_query_key(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.queue_prefetch("   why does the loader skip my provider   ")
    assert provider._drain_for_test()

    provider.prefetch("why does the loader skip my provider")

    assert count_requests(fake_engram, "GET", "/search") == 1


@pytest.mark.parametrize("trivial", ["", "  ", "ok", "/reset", "thanks"])
def test_queue_prefetch_skips_trivial_queries(
    fake_engram: FakeEngram, provider_factory: Any, trivial: str
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)

    provider.queue_prefetch(trivial)
    assert provider._drain_for_test()

    assert fake_engram.requests == []


def test_queue_prefetch_is_a_no_op_when_disabled(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url, enabled=False)

    provider.queue_prefetch("a real question about the loader")
    assert provider._drain_for_test()

    assert fake_engram.requests == []


def test_queue_prefetch_does_not_raise_with_no_server_listening(provider_factory: Any) -> None:
    provider = provider_factory(f"http://127.0.0.1:{unused_port()}")

    provider.queue_prefetch("a real question about the loader")

    assert provider._drain_for_test()
    assert provider.prefetch("a real question about the loader") == ""


def test_invalidating_the_cache_forces_a_fresh_search(
    fake_engram: FakeEngram, provider_factory: Any
) -> None:
    wire_engram(fake_engram)
    provider = provider_factory(fake_engram.base_url)
    provider.queue_prefetch("why does the loader skip my provider")
    assert provider._drain_for_test()

    provider._invalidate_prefetch_cache()
    provider.prefetch("why does the loader skip my provider")

    assert count_requests(fake_engram, "GET", "/search") == 2


# -- 9. shutdown and worker ---------------------------------------------------


def test_shutdown_is_safe_when_the_worker_never_started() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1"))

    provider.shutdown()
    provider.shutdown()


def test_shutdown_after_initialize_stops_the_worker(fake_engram: FakeEngram) -> None:
    wire_engram(fake_engram)
    provider = RecallProvider(config=make_config(fake_engram.base_url))
    provider.initialize(session_id="sess-1", hermes_home="/tmp/h", platform="cli")
    assert provider._drain_for_test()

    provider.shutdown()

    assert provider._worker is None
    # And a second call is still a no-op.
    provider.shutdown()


def test_submitting_after_shutdown_is_dropped(fake_engram: FakeEngram) -> None:
    wire_engram(fake_engram)
    provider = RecallProvider(config=make_config(fake_engram.base_url))
    provider.initialize(session_id="sess-1", hermes_home="/tmp/h", platform="cli")
    assert provider._drain_for_test()
    provider.shutdown()
    fake_engram.reset()

    provider.queue_prefetch("a real question about the loader")

    assert provider._worker is None
    assert fake_engram.requests == []


def test_drain_for_test_returns_true_on_an_idle_provider() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1"))
    assert provider._drain_for_test(timeout=1.0) is True


# -- 10. tools ----------------------------------------------------------------


def test_get_tool_schemas_is_stable_and_pure() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1"))

    first = provider.get_tool_schemas()
    second = provider.get_tool_schemas()

    assert len(first) == 10
    assert first == second
    # A fresh copy each call: mutating one result must not leak into the next.
    first[0]["name"] = "mutated"
    assert provider.get_tool_schemas()[0]["name"] == "mem_search"


def test_get_tool_schemas_makes_no_request(fake_engram: FakeEngram) -> None:
    # The host calls it at least three times during registration alone.
    wire_engram(fake_engram)
    provider = RecallProvider(config=make_config(fake_engram.base_url))

    for _ in range(3):
        assert len(provider.get_tool_schemas()) == 10

    assert fake_engram.requests == []


def test_get_tool_schemas_is_empty_when_disabled() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1", enabled=False))
    assert provider.get_tool_schemas() == []


# -- 11. every hook is a real override ----------------------------------------


@pytest.mark.parametrize(
    "hook",
    [
        "on_turn_start",
        "sync_turn",
        "on_delegation",
        "on_session_end",
        "on_session_switch",
        "on_pre_compress",
        "get_tool_schemas",
        "handle_tool_call",
    ],
)
def test_capture_and_lifecycle_hooks_are_implemented(hook: str) -> None:
    """Every hook T4-T6 own must be a real override, not the inherited default."""
    assert hook in vars(RecallProvider), f"{hook} is still the inherited no-op"


def test_pre_compress_checkpoint_api_version_is_one() -> None:
    assert RecallProvider.pre_compress_checkpoint_api_version == 1
