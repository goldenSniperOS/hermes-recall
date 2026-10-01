"""Behaviour of ``hermes_recall.client.EngramClient`` against a fake Engram server.

Nothing here touches a real Engram: every test either points the client at a
:class:`~fake_engram.FakeEngram` on an ephemeral port, or at a port with nothing
listening on it.
"""

from __future__ import annotations

import inspect
import socket
import time
import typing
from typing import Any

import pytest
from fake_engram import Delay, FakeEngram, Malformed, RecordedRequest

from hermes_recall import client as client_module
from hermes_recall.client import EngramClient
from hermes_recall.config import RecallConfig

# -- helpers ------------------------------------------------------------------


def make_config(url: str, *, auth_token: str = "") -> RecallConfig:
    """A fully specified config, so no test depends on the ambient environment."""
    return RecallConfig(
        url=url,
        project_override="",
        recall_limit=5,
        recall_max_chars=2500,
        enabled=True,
        auth_token=auth_token,
    )


def client_for(fake: FakeEngram, *, auth_token: str = "") -> EngramClient:
    """A client pointed at ``fake``."""
    return EngramClient(make_config(fake.base_url, auth_token=auth_token))


def unused_port() -> int:
    """Bind port 0, read the port, release it — nothing will be listening there."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


HEALTH_PAYLOAD = {
    "status": "ok",
    "service": "engram",
    "version": "2.0.0",
    "instance_id": "abc123",
}


# -- 1. health and base_url ---------------------------------------------------


def test_health_returns_the_payload(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/health", (200, HEALTH_PAYLOAD))

    client = client_for(fake_engram)
    assert client.health() == HEALTH_PAYLOAD
    assert client.last_error_text() == ""


def test_base_url_has_no_trailing_slash_and_requests_still_land(
    fake_engram: FakeEngram,
) -> None:
    fake_engram.route("GET", "/health", (200, HEALTH_PAYLOAD))
    client = EngramClient(make_config(fake_engram.base_url + "///"))

    assert client.base_url == fake_engram.base_url
    assert not client.base_url.endswith("/")
    assert client.health() == HEALTH_PAYLOAD
    assert fake_engram.find("GET", "/health") is not None


# -- 2. current_project -------------------------------------------------------


def test_current_project_sends_cwd_and_returns_the_dict(fake_engram: FakeEngram) -> None:
    payload = {
        "project": "hermes-recall",
        "project_source": "cwd",
        "project_path": "/Users/dev/hermes-recall",
        "cwd": "/Users/dev/hermes-recall",
        "available_projects": ["hermes-recall"],
    }
    fake_engram.route("GET", "/project/current", (200, payload))

    assert client_for(fake_engram).current_project("/Users/dev/hermes-recall") == payload
    request = fake_engram.find("GET", "/project/current")
    assert request is not None
    assert request.query_one("cwd") == "/Users/dev/hermes-recall"


def test_ambiguous_project_answer_is_returned_unchanged(fake_engram: FakeEngram) -> None:
    # A 200 with an empty project plus an error_hint is Engram telling us it
    # cannot decide, not a transport failure. It must survive intact.
    payload = {
        "project": "",
        "project_source": "",
        "project_path": "",
        "cwd": "/tmp/x",
        "available_projects": ["alpha", "beta"],
        "error_hint": "ambiguous project: alpha, beta",
    }
    fake_engram.route("GET", "/project/current", (200, payload))

    client = client_for(fake_engram)
    assert client.current_project("/tmp/x") == payload
    assert client.last_error_text() == ""


# -- 3-5. search request shape ------------------------------------------------


def test_search_request_shape(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/search", (200, []))

    client_for(fake_engram).search("auth bug", project="hermes-recall", limit=3)

    request = fake_engram.find("GET", "/search")
    assert request is not None
    assert request.query_one("q") == "auth bug"
    assert request.query_one("project") == "hermes-recall"
    assert request.query_one("limit") == "3"
    assert request.query_one("match_mode") == "any"
    # `project` and `all_projects` together is a 400 on the server.
    assert "all_projects" not in request.query
    # Blank optionals are dropped rather than sent empty: the server rejects a
    # blank `project` outright and uses absence to pick its own defaults.
    assert "type" not in request.query
    assert "scope" not in request.query


def test_search_all_projects_replaces_project(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/search", (200, []))

    client_for(fake_engram).search("auth bug", project="hermes-recall", all_projects=True)

    request = fake_engram.find("GET", "/search")
    assert request is not None
    assert request.query_one("all_projects") == "true"
    assert "project" not in request.query


def test_blank_search_makes_no_request(fake_engram: FakeEngram) -> None:
    client = client_for(fake_engram)

    assert client.search("   ") == []
    assert fake_engram.requests == []
    assert client.last_error_text() != ""


@pytest.mark.parametrize("bad_mode", ["fuzzy", "ALL_OF_THEM", "", "   ", "anyy"])
def test_search_drops_an_invalid_match_mode(fake_engram: FakeEngram, bad_mode: str) -> None:
    fake_engram.route("GET", "/search", (200, []))

    client_for(fake_engram).search("auth bug", match_mode=bad_mode)

    request = fake_engram.find("GET", "/search")
    assert request is not None
    assert "match_mode" not in request.query


@pytest.mark.parametrize(("raw", "expected"), [("ALL", "all"), ("  Any  ", "any")])
def test_search_normalizes_a_valid_match_mode(
    fake_engram: FakeEngram, raw: str, expected: str
) -> None:
    fake_engram.route("GET", "/search", (200, []))

    client_for(fake_engram).search("auth bug", match_mode=raw)

    request = fake_engram.find("GET", "/search")
    assert request is not None
    assert request.query_one("match_mode") == expected


# -- 6. search failure modes --------------------------------------------------


@pytest.mark.parametrize(
    "route",
    [
        (200, {"not": "a list"}),
        (200, "a bare string"),
        (200, None),
        (500, {"error": "boom"}),
        (400, {"error": "q parameter is required"}),
        Malformed(),
    ],
    ids=["dict-body", "string-body", "empty-body", "500", "400", "malformed-bytes"],
)
def test_search_returns_empty_list_on_any_bad_answer(fake_engram: FakeEngram, route: Any) -> None:
    fake_engram.route("GET", "/search", route)

    assert client_for(fake_engram).search("auth bug") == []


def test_search_returns_the_list_on_success(fake_engram: FakeEngram) -> None:
    hits = [{"id": 1, "title": "Fixed auth"}, {"id": 2, "title": "Auth notes"}]
    fake_engram.route("GET", "/search", (200, hits))

    assert client_for(fake_engram).search("auth") == hits


# -- 7. add_observation -------------------------------------------------------


def test_add_observation_posts_the_right_body(fake_engram: FakeEngram) -> None:
    fake_engram.route("POST", "/observations", (201, {"id": 761, "status": "saved"}))

    result = client_for(fake_engram).add_observation(
        "sess-1",
        "Fixed auth bug",
        "**What**: patched the middleware",
        type_="bugfix",
        project="hermes-recall",
    )

    assert result == {"id": 761, "status": "saved"}
    request = fake_engram.find("POST", "/observations")
    assert request is not None
    assert request.json_body == {
        "session_id": "sess-1",
        "type": "bugfix",
        "title": "Fixed auth bug",
        "content": "**What**: patched the middleware",
        "project": "hermes-recall",
        "scope": "project",
    }
    # Blank optionals are omitted entirely.
    assert "topic_key" not in request.json_body
    assert "tool_name" not in request.json_body


def test_add_observation_sends_every_optional_when_given(fake_engram: FakeEngram) -> None:
    fake_engram.route("POST", "/observations", (201, {"id": 1, "status": "saved"}))

    client_for(fake_engram).add_observation(
        "sess-1",
        "Title",
        "Content",
        type_="decision",
        project="proj",
        scope="personal",
        topic_key="architecture/auth-model",
        tool_name="mem_save",
    )

    request = fake_engram.find("POST", "/observations")
    assert request is not None
    assert request.json_body == {
        "session_id": "sess-1",
        "type": "decision",
        "title": "Title",
        "content": "Content",
        "project": "proj",
        "scope": "personal",
        "topic_key": "architecture/auth-model",
        "tool_name": "mem_save",
    }


@pytest.mark.parametrize(
    ("session_id", "title", "content"),
    [
        ("", "Title", "Content"),
        ("   ", "Title", "Content"),
        ("sess-1", "", "Content"),
        ("sess-1", "   ", "Content"),  # whitespace-only title: a server 400
        ("sess-1", "Title", ""),
        ("sess-1", "Title", "   "),
    ],
)
def test_add_observation_refuses_blank_required_fields_without_a_request(
    fake_engram: FakeEngram, session_id: str, title: str, content: str
) -> None:
    client = client_for(fake_engram)

    assert client.add_observation(session_id, title, content) is None
    assert fake_engram.requests == []
    assert client.last_error_text() != ""


# -- 8. update_observation ----------------------------------------------------


def test_update_observation_sends_only_the_keys_passed(fake_engram: FakeEngram) -> None:
    updated = {"id": 761, "title": "New title", "content": "body"}
    fake_engram.route("PATCH", "/observations/761", (200, updated))

    result = client_for(fake_engram).update_observation(761, title="New title")

    assert result == updated
    request = fake_engram.find("PATCH", "/observations/761")
    assert request is not None
    assert request.json_body == {"title": "New title"}


def test_update_observation_keeps_an_explicit_empty_string(fake_engram: FakeEngram) -> None:
    # The server's patch params are pointers, so "" means "set it to empty" and
    # must reach the wire; only `None` means "leave it alone".
    fake_engram.route("PATCH", "/observations/5", (200, {"id": 5}))

    client_for(fake_engram).update_observation(5, topic_key="", title=None)

    request = fake_engram.find("PATCH", "/observations/5")
    assert request is not None
    assert request.json_body == {"topic_key": ""}


def test_update_observation_ignores_unknown_keys(fake_engram: FakeEngram) -> None:
    fake_engram.route("PATCH", "/observations/5", (200, {"id": 5}))

    client_for(fake_engram).update_observation(5, content="body", nonsense="x", session_id="s")

    request = fake_engram.find("PATCH", "/observations/5")
    assert request is not None
    assert request.json_body == {"content": "body"}


def test_update_observation_with_no_fields_makes_no_request(fake_engram: FakeEngram) -> None:
    client = client_for(fake_engram)

    assert client.update_observation(761) is None
    assert fake_engram.requests == []
    assert client.last_error_text() != ""


def test_update_observation_with_only_none_fields_makes_no_request(
    fake_engram: FakeEngram,
) -> None:
    client = client_for(fake_engram)

    assert client.update_observation(761, title=None, content=None) is None
    assert fake_engram.requests == []


# -- 9. passive_capture -------------------------------------------------------


def test_passive_capture_posts_and_returns_the_counts(fake_engram: FakeEngram) -> None:
    counts = {"extracted": 3, "saved": 2, "duplicates": 1}
    fake_engram.route("POST", "/observations/passive", (200, counts))

    result = client_for(fake_engram).passive_capture(
        "sess-1", "## Key Learnings:\n1. x", project="hermes-recall", source="session-end"
    )

    assert result == counts
    request = fake_engram.find("POST", "/observations/passive")
    assert request is not None
    assert request.json_body == {
        "session_id": "sess-1",
        "content": "## Key Learnings:\n1. x",
        "project": "hermes-recall",
        "source": "session-end",
    }


def test_passive_capture_omits_blank_optionals(fake_engram: FakeEngram) -> None:
    fake_engram.route("POST", "/observations/passive", (200, {"extracted": 0}))

    client_for(fake_engram).passive_capture("sess-1", "content")

    request = fake_engram.find("POST", "/observations/passive")
    assert request is not None
    assert request.json_body == {"session_id": "sess-1", "content": "content"}


# -- 10. context and compaction_context ---------------------------------------


def test_context_returns_the_context_string(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/context", (200, {"context": "## Recent\n- did a thing"}))

    assert client_for(fake_engram).context(project="hermes-recall") == "## Recent\n- did a thing"
    request = fake_engram.find("GET", "/context")
    assert request is not None
    assert request.query_one("project") == "hermes-recall"
    assert "scope" not in request.query


def test_compaction_context_returns_the_context_string(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/context/compaction", (200, {"context": "earlier work"}))

    assert client_for(fake_engram).compaction_context("sess-1") == "earlier work"
    request = fake_engram.find("GET", "/context/compaction")
    assert request is not None
    assert request.query_one("session_id") == "sess-1"


@pytest.mark.parametrize(
    "route",
    [
        (404, {"error": "session not found"}),
        (500, {"error": "boom"}),
        (200, {"no_context_key": "x"}),
        (200, {"context": 42}),
        (200, []),
        Malformed(),
    ],
    ids=["404", "500", "missing-key", "non-string", "list-body", "malformed-bytes"],
)
def test_context_endpoints_return_empty_string_on_failure(
    fake_engram: FakeEngram, route: Any
) -> None:
    fake_engram.route("GET", "/context", route)
    fake_engram.route("GET", "/context/compaction", route)

    client = client_for(fake_engram)
    assert client.context() == ""
    assert client.compaction_context("sess-1") == ""


def test_blank_compaction_session_id_makes_no_request(fake_engram: FakeEngram) -> None:
    client = client_for(fake_engram)

    assert client.compaction_context("  ") == ""
    assert fake_engram.requests == []
    assert client.last_error_text() != ""


# -- 11. recent_observations --------------------------------------------------


def test_recent_observations_returns_a_list(fake_engram: FakeEngram) -> None:
    rows = [{"id": 2, "title": "b"}, {"id": 1, "title": "a"}]
    fake_engram.route("GET", "/observations/recent", (200, rows))

    assert client_for(fake_engram).recent_observations(project="proj", limit=2) == rows
    request = fake_engram.find("GET", "/observations/recent")
    assert request is not None
    assert request.query_one("project") == "proj"
    assert request.query_one("limit") == "2"


@pytest.mark.parametrize(
    "route",
    [(500, {"error": "boom"}), (200, {"not": "a list"}), (404, {"error": "nope"}), Malformed()],
    ids=["500", "dict-body", "404", "malformed-bytes"],
)
def test_recent_observations_returns_empty_list_on_failure(
    fake_engram: FakeEngram, route: Any
) -> None:
    fake_engram.route("GET", "/observations/recent", route)

    assert client_for(fake_engram).recent_observations() == []


# -- sessions, prompts, timeline, stats, get_observation ----------------------


def test_create_session_posts_project_owned_ownership(fake_engram: FakeEngram) -> None:
    fake_engram.route("POST", "/sessions", (201, {"id": "sess-1", "status": "created"}))

    result = client_for(fake_engram).create_session("sess-1", "hermes-recall")

    assert result == {"id": "sess-1", "status": "created"}
    request = fake_engram.find("POST", "/sessions")
    assert request is not None
    assert request.json_body == {
        "id": "sess-1",
        "project": "hermes-recall",
        "ownership_mode": "project_owned",
    }
    assert "directory" not in request.json_body


def test_create_session_includes_a_non_blank_directory(fake_engram: FakeEngram) -> None:
    fake_engram.route("POST", "/sessions", (201, {"id": "sess-1", "status": "created"}))

    client_for(fake_engram).create_session("sess-1", "proj", directory="/Users/dev/proj")

    request = fake_engram.find("POST", "/sessions")
    assert request is not None
    assert request.json_body["directory"] == "/Users/dev/proj"


def test_end_session_quotes_the_id_in_the_path(fake_engram: FakeEngram) -> None:
    # A session id can carry slashes (worktree-derived ids do); unquoted they
    # would change the route. The server sees — and routes on — the encoded form,
    # which is exactly what the recorded path asserts here.
    fake_engram.route(
        "POST", "/sessions/a%2Fb%20c/end", (200, {"id": "a/b c", "status": "completed"})
    )

    result = client_for(fake_engram).end_session("a/b c", summary="done")

    assert result == {"id": "a/b c", "status": "completed"}
    request = fake_engram.find("POST", "/sessions/a%2Fb%20c/end")
    assert request is not None
    assert request.json_body == {"summary": "done"}
    # The raw slash never reaches the path, so the route cannot be hijacked.
    assert fake_engram.find("POST", "/sessions/a/b c/end") is None


def test_get_observation_returns_the_row(fake_engram: FakeEngram) -> None:
    row = {"id": 761, "title": "Fixed auth", "content": "body"}
    fake_engram.route("GET", "/observations/761", (200, row))

    assert client_for(fake_engram).get_observation(761) == row


def test_get_observation_returns_none_on_404(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/observations/9", (404, {"error": "observation not found"}))

    client = client_for(fake_engram)
    assert client.get_observation(9) is None
    assert "404" in client.last_error_text()
    assert "observation not found" in client.last_error_text()


def test_add_prompt_posts_session_and_content(fake_engram: FakeEngram) -> None:
    fake_engram.route("POST", "/prompts", (201, {"id": 12, "status": "saved"}))

    result = client_for(fake_engram).add_prompt("sess-1", "what did we do?", project="proj")

    assert result == {"id": 12, "status": "saved"}
    request = fake_engram.find("POST", "/prompts")
    assert request is not None
    assert request.json_body == {
        "session_id": "sess-1",
        "content": "what did we do?",
        "project": "proj",
    }


@pytest.mark.parametrize(
    ("session_id", "content"), [("", "x"), ("  ", "x"), ("sess-1", ""), ("sess-1", "   ")]
)
def test_add_prompt_refuses_blank_required_fields_without_a_request(
    fake_engram: FakeEngram, session_id: str, content: str
) -> None:
    client = client_for(fake_engram)

    assert client.add_prompt(session_id, content) is None
    assert fake_engram.requests == []
    assert client.last_error_text() != ""


def test_timeline_sends_observation_id_before_and_after(fake_engram: FakeEngram) -> None:
    payload = {"focus": {"id": 761}, "before": [], "after": []}
    fake_engram.route("GET", "/timeline", (200, payload))

    assert client_for(fake_engram).timeline(761, before=2, after=3) == payload
    request = fake_engram.find("GET", "/timeline")
    assert request is not None
    assert request.query_one("observation_id") == "761"
    assert request.query_one("before") == "2"
    assert request.query_one("after") == "3"


def test_timeline_keeps_a_zero_neighbour_count(fake_engram: FakeEngram) -> None:
    # 0 is a real instruction ("no neighbours"), not an absent value, so it must
    # not be dropped the way "" is.
    fake_engram.route("GET", "/timeline", (200, {"focus": {"id": 1}}))

    client_for(fake_engram).timeline(1, before=0, after=0)

    request = fake_engram.find("GET", "/timeline")
    assert request is not None
    assert request.query_one("before") == "0"
    assert request.query_one("after") == "0"


def test_stats_sends_project_or_all_projects_but_never_both(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/stats", (200, {"observations": 10}))
    client = client_for(fake_engram)

    assert client.stats(project="proj") == {"observations": 10}
    first = fake_engram.find("GET", "/stats")
    assert first is not None
    assert first.query_one("project") == "proj"
    assert "all_projects" not in first.query

    fake_engram.reset()
    client.stats(project="proj", all_projects=True)
    second = fake_engram.find("GET", "/stats")
    assert second is not None
    assert second.query_one("all_projects") == "true"
    assert "project" not in second.query


# -- 12. fail-soft sweep ------------------------------------------------------

# Every public endpoint method, with arguments good enough to reach the network.
# Methods that guard locally (search, add_observation, add_prompt,
# compaction_context, update_observation) get valid arguments on purpose, so the
# sweep exercises the transport rather than the local short-circuit.
SWEEP_CALLS: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {
    "health": ((), {}),
    "current_project": (("/tmp/x",), {}),
    "create_session": (("sess-1", "proj"), {}),
    "end_session": (("sess-1",), {"summary": "done"}),
    "add_observation": (("sess-1", "Title", "Content"), {}),
    "update_observation": ((761,), {"title": "New"}),
    "get_observation": ((761,), {}),
    "recent_observations": ((), {}),
    "passive_capture": (("sess-1", "content"), {}),
    "search": (("query",), {}),
    "context": ((), {}),
    "compaction_context": (("sess-1",), {}),
    "add_prompt": (("sess-1", "content"), {}),
    "timeline": ((761,), {}),
    "stats": ((), {}),
}

# Public callables that are introspection, not HTTP, and so are not swept.
NON_ENDPOINT_METHODS = frozenset({"last_error_text"})


def public_endpoint_methods() -> dict[str, Any]:
    """Every public method of ``EngramClient`` that is meant to make a request."""
    return {
        name: member
        for name, member in inspect.getmembers(EngramClient, predicate=inspect.isfunction)
        if not name.startswith("_") and name not in NON_ENDPOINT_METHODS
    }


def documented_empty_value(method: Any) -> Any:
    """Derive a method's documented failure value from its return annotation.

    Reading the annotation instead of hardcoding a table is what makes the sweep
    hold for methods that do not exist yet: a new method is swept against its own
    declared contract, with no second list to keep in sync.
    """
    annotation = typing.get_type_hints(method).get("return")
    if annotation is str:
        return ""
    if typing.get_origin(annotation) is list:
        return []
    args = typing.get_args(annotation)
    if type(None) in args:
        return None
    raise AssertionError(
        f"{method.__name__} has return annotation {annotation!r}, which declares no "
        "fail-soft empty value. Annotate it `-> X | None`, `-> list[...]` or `-> str`."
    )


def test_sweep_table_covers_every_public_method() -> None:
    """A new public method must be added to the sweep, not silently skipped."""
    assert set(public_endpoint_methods()) == set(SWEEP_CALLS), (
        "SWEEP_CALLS is out of sync with EngramClient's public surface; "
        "a new method must prove it is fail-soft too."
    )


def test_every_public_method_is_fail_soft_with_no_server_listening() -> None:
    client = EngramClient(make_config(f"http://127.0.0.1:{unused_port()}"))
    methods = public_endpoint_methods()

    for name, method in methods.items():
        args, kwargs = SWEEP_CALLS[name]
        expected = documented_empty_value(method)
        try:
            result = getattr(client, name)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the whole point of the test
            raise AssertionError(f"{name}() raised {exc!r}; it must fail soft") from exc
        assert result == expected, f"{name}() returned {result!r}, expected {expected!r}"
        assert client.last_error_text() != "", f"{name}() recorded no error"


def test_every_public_method_is_fail_soft_against_a_500(fake_engram: FakeEngram) -> None:
    # A server that answers but always errors is a different failure class from
    # one that is absent; both have to degrade the same way.
    fake_engram.default_response = (500, {"error": "boom"})
    client = client_for(fake_engram)

    for name, method in public_endpoint_methods().items():
        args, kwargs = SWEEP_CALLS[name]
        expected = documented_empty_value(method)
        result = getattr(client, name)(*args, **kwargs)
        assert result == expected, f"{name}() returned {result!r}, expected {expected!r}"


def test_every_public_method_is_fail_soft_against_malformed_bytes(
    fake_engram: FakeEngram,
) -> None:
    fake_engram.default_response = Malformed()  # type: ignore[assignment]
    client = client_for(fake_engram)

    for name, method in public_endpoint_methods().items():
        args, kwargs = SWEEP_CALLS[name]
        expected = documented_empty_value(method)
        result = getattr(client, name)(*args, **kwargs)
        assert result == expected, f"{name}() returned {result!r}, expected {expected!r}"


# -- 13. timeout --------------------------------------------------------------


def test_read_timeout_gives_up_and_records_an_error(
    fake_engram: FakeEngram, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The constants are read through the module namespace at call time, so this
    # monkeypatch takes effect; binding them as default arguments would not.
    monkeypatch.setattr(client_module, "READ_TIMEOUT_S", 0.2)
    fake_engram.route("GET", "/search", Delay(2.0, (200, [{"id": 1}])))

    client = client_for(fake_engram)
    started = time.monotonic()
    result = client.search("auth")
    elapsed = time.monotonic() - started

    assert result == []
    assert client.last_error_text() != ""
    # Proof it actually gave up rather than waiting out the delay.
    assert elapsed < 1.5, f"client waited {elapsed:.2f}s; it should abandon the call"


def test_write_timeout_gives_up_and_records_an_error(
    fake_engram: FakeEngram, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client_module, "WRITE_TIMEOUT_S", 0.2)
    fake_engram.route("POST", "/observations", Delay(2.0, (201, {"id": 1, "status": "saved"})))

    client = client_for(fake_engram)
    started = time.monotonic()
    result = client.add_observation("sess-1", "Title", "Content")
    elapsed = time.monotonic() - started

    assert result is None
    assert client.last_error_text() != ""
    assert elapsed < 1.5, f"client waited {elapsed:.2f}s; it should abandon the call"


def test_timeouts_stay_well_below_the_host_prefetch_budget() -> None:
    # Hermes joins an external prefetch for 8s and then skips the provider on
    # every later turn until the stuck call returns, so a client timeout at or
    # near 8s would cost the whole session, not one turn.
    assert client_module.READ_TIMEOUT_S < 8.0
    assert client_module.WRITE_TIMEOUT_S < 8.0


# -- 14. Authorization header -------------------------------------------------


def test_authorization_header_sent_when_token_is_set(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/health", (200, HEALTH_PAYLOAD))

    client_for(fake_engram, auth_token="s3cret").health()

    request = fake_engram.last_request
    assert request is not None
    assert request.authorization == "Bearer s3cret"


def test_authorization_header_absent_when_token_is_unset(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/health", (200, HEALTH_PAYLOAD))

    client_for(fake_engram).health()

    request = fake_engram.last_request
    assert request is not None
    assert request.authorization is None


def test_authorization_header_sent_on_writes_too(fake_engram: FakeEngram) -> None:
    fake_engram.route("POST", "/observations", (201, {"id": 1, "status": "saved"}))

    client_for(fake_engram, auth_token="s3cret").add_observation("s", "T", "C")

    request = fake_engram.last_request
    assert request is not None
    assert request.authorization == "Bearer s3cret"


# -- 15. last_error lifecycle -------------------------------------------------


def test_last_error_is_cleared_by_a_success_after_a_failure(fake_engram: FakeEngram) -> None:
    responses: list[Any] = [(500, {"error": "boom"}), (200, HEALTH_PAYLOAD)]

    def answer(_request: RecordedRequest) -> Any:
        return responses.pop(0)

    fake_engram.route("GET", "/health", answer)
    client = client_for(fake_engram)

    assert client.health() is None
    assert client.last_error_text() != ""
    assert client.last_error == client.last_error_text()

    assert client.health() == HEALTH_PAYLOAD
    assert client.last_error_text() == ""


def test_last_error_records_status_and_server_message_only(fake_engram: FakeEngram) -> None:
    fake_engram.route("POST", "/observations", (409, {"error": "session ownership conflict"}))

    client = client_for(fake_engram)
    secret_content = "the user's private memory text"
    assert client.add_observation("sess-1", "Title", secret_content) is None

    error = client.last_error_text()
    assert "409" in error
    assert "/observations" in error
    assert "session ownership conflict" in error
    # The recorded reason must never leak user content.
    assert secret_content not in error


def test_failures_are_logged_at_debug_only(
    fake_engram: FakeEngram, caplog: pytest.LogCaptureFixture
) -> None:
    # The provider is fail-soft: an Engram that is simply not running must not
    # print a warning on every single turn.
    fake_engram.route("GET", "/health", (500, {"error": "boom"}))

    with caplog.at_level("DEBUG", logger="hermes_recall.client"):
        assert client_for(fake_engram).health() is None

    records = [r for r in caplog.records if r.name == "hermes_recall.client"]
    assert records, "the failure should still be logged"
    assert all(r.levelname == "DEBUG" for r in records), [r.levelname for r in records]


def test_client_without_an_explicit_config_loads_one(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = make_config("http://sentinel:1")
    monkeypatch.setattr(client_module, "load_config", lambda: sentinel)

    assert EngramClient().base_url == "http://sentinel:1"
