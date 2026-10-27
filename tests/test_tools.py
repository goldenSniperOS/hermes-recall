"""The ``mem_*`` tools: request shapes, JSON results, status strings, state and fail-soft.

Every test talks to a :class:`~fake_engram.FakeEngram` on an ephemeral port or to a
port with nothing listening — never the user's real Engram on 127.0.0.1:7437.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_engram import Delay, FakeEngram, Malformed, RecordedRequest

from hermes_recall import provider as provider_module
from hermes_recall import tools as tools_module
from hermes_recall.config import RecallConfig
from hermes_recall.provider import RecallProvider
from hermes_recall.state import read_state, state_path

PROJECT = "hermes-recall"
SESSION = "sess-1"

EXPECTED_TOOLS = [
    "mem_search",
    "mem_save",
    "mem_update",
    "mem_get_observation",
    "mem_context",
    "mem_session_summary",
    "mem_timeline",
    "mem_judge",
    "mem_pin",
    "mem_unpin",
    "mem_consolidate",
    "mem_judge_many",
    "mem_session_finalize",
]

OBSERVATION = {
    "id": 761,
    "type": "decision",
    "title": "Use stdlib urllib",
    "content": "**What**: chose urllib\n**Why**: no deps",
    "project": PROJECT,
    "scope": "project",
    "topic_key": "",
    "created_at": "2026-01-02 03:04:05",
}

# What `wire` detects carries no project_source, so capture evidence is unknown.
UNKNOWN_PROVENANCE = {"source": "unknown", "directory": provider_module._working_directory()}

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


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def echo_session(request: RecordedRequest) -> tuple[int, Any]:
    """Acknowledge ``POST /sessions`` with the id it was sent, as Engram 2.0.0 does."""
    return (201, {"id": request.json_body["id"], "status": "created"})


def wire(fake: FakeEngram) -> None:
    fake.route("GET", "/project/current", (200, {"project": PROJECT}))
    fake.route("POST", "/sessions", echo_session)
    fake.route("POST", "/observations", (201, {"id": 900, "status": "saved"}))
    fake.route("GET", "/observations/761", (200, OBSERVATION))
    fake.route("PATCH", "/observations/761", (200, {**OBSERVATION, "title": "New"}))
    fake.route("PUT", "/observations/761/pin", (200, {"id": 761, "pinned": True}))
    fake.route("DELETE", "/observations/761/pin", (200, {"id": 761, "pinned": False}))
    fake.route("GET", "/search", (200, [OBSERVATION]))
    fake.route("GET", "/context", (200, {"context": "## Recent\n- did a thing"}))
    fake.route("GET", "/timeline", (200, {"focus": OBSERVATION, "before": [], "after": []}))
    fake.route(
        "POST",
        "/conflicts/judge",
        (200, {"relation": {"judgment_id": "rel-1", "relation": "supersedes"}}),
    )


@pytest.fixture
def make_provider(tmp_path: Path) -> Iterator[Any]:
    built: list[RecallProvider] = []

    def factory(url: str, *, agent_context: str = "primary", **config_over: Any):
        provider = RecallProvider(config=make_config(url, **config_over))
        provider.initialize(
            SESSION,
            hermes_home=str(tmp_path),
            platform="cli",
            agent_context=agent_context,
        )
        assert provider._drain_for_test()
        built.append(provider)
        return provider

    yield factory
    for provider in built:
        provider.shutdown()


def call(provider: RecallProvider, name: str, **args: Any) -> dict[str, Any]:
    raw = provider.handle_tool_call(name, args)
    assert isinstance(raw, str)
    return json.loads(raw)


def only(fake: FakeEngram, method: str, path: str) -> RecordedRequest:
    matches = [r for r in fake.requests if r.method == method and r.path == path]
    assert len(matches) == 1, [(r.method, r.path) for r in fake.requests]
    return matches[0]


# -- schemas ------------------------------------------------------------------


def test_schemas_have_the_expected_names_in_order() -> None:
    schemas = tools_module.get_tool_schemas()
    assert [s["name"] for s in schemas] == EXPECTED_TOOLS
    assert tools_module.TOOL_NAMES == frozenset(EXPECTED_TOOLS)


def test_schemas_are_bare_function_schemas_with_short_descriptions() -> None:
    for schema in tools_module.get_tool_schemas():
        assert set(schema) == {"name", "description", "parameters"}
        assert len(schema["description"]) <= 80, schema["name"]
        params = schema["parameters"]
        assert params["type"] == "object"
        assert set(params["required"]) <= set(params["properties"])


def test_total_schema_payload_stays_compact() -> None:
    # Token cost is the reason this tool set exists; keep it honest.
    # L3: raised from 4500 for the three Engram 3.6 workflow tools.
    assert len(json.dumps(tools_module.get_tool_schemas())) < 5600


def test_write_tools_are_a_subset_of_the_tools() -> None:
    assert tools_module.WRITE_TOOLS <= tools_module.TOOL_NAMES
    assert "mem_search" not in tools_module.WRITE_TOOLS


# -- mem_search ---------------------------------------------------------------


def test_mem_search_request_and_result(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_search", query="urllib", limit=3, type="decision")

    request = only(fake_engram, "GET", "/search")
    assert request.query_one("q") == "urllib"
    assert request.query_one("project") == PROJECT
    assert request.query_one("limit") == "3"
    assert request.query_one("match_mode") == "any"
    assert request.query_one("type") == "decision"
    assert result["count"] == 1
    assert result["project"] == PROJECT
    hit = result["results"][0]
    assert hit["id"] == 761
    assert hit["title"] == "Use stdlib urllib"
    assert "content" not in hit
    assert hit["snippet"].startswith("**What**: chose urllib")


def test_mem_search_renders_compact_hits_with_their_pin(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    compact = {
        "id": 761,
        "sync_id": "obs-1",
        "title": "Use stdlib urllib",
        "type": "decision",
        "state": "active",
        "scope": "project",
        "pinned": True,
        "preview": "**What**: chose\nurllib",
        "truncated": True,
        "project": PROJECT,
    }
    unpinned = {**compact, "id": 762, "pinned": False}
    fake_engram.route(
        "GET", "/search", (200, {"result": "Found 2 memories.", "results": [compact, unpinned]})
    )
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_search", query="urllib")

    assert only(fake_engram, "GET", "/search").query_one("response_format") == "compact"
    assert result["count"] == 2
    first, second = result["results"]
    assert first == {
        "id": 761,
        "type": "decision",
        "title": "Use stdlib urllib",
        "project": PROJECT,
        "scope": "project",
        "pinned": True,
        "snippet": "**What**: chose urllib",
    }
    assert "pinned" not in second


def test_mem_search_on_a_legacy_bare_array_server_still_works(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    # Engram < 3.6 ignores response_format and answers the full bare array.
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_search", query="urllib")

    assert only(fake_engram, "GET", "/search").query_one("response_format") == "compact"
    assert result["results"][0]["snippet"].startswith("**What**: chose urllib")
    assert "pinned" not in result["results"][0]


def test_mem_search_all_projects_drops_project_and_clamps_limit(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    call(provider, "mem_search", query="x", all_projects=True, limit=999)

    request = only(fake_engram, "GET", "/search")
    assert "project" not in request.query
    assert request.query_one("all_projects") == "true"
    assert request.query_one("limit") == str(tools_module.SEARCH_LIMIT_MAX)


def test_mem_search_snippet_is_capped(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    fake_engram.route("GET", "/search", (200, [{**OBSERVATION, "content": "x" * 5000}]))
    provider = make_provider(fake_engram.base_url)

    hit = call(provider, "mem_search", query="x")["results"][0]

    assert len(hit["snippet"]) <= tools_module.SEARCH_SNIPPET_MAX_CHARS + 3


def test_mem_search_requires_a_query(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    assert "query" in call(provider, "mem_search")["error"]
    assert "match_mode" in call(provider, "mem_search", query="x", match_mode="fuzzy")["error"]
    assert fake_engram.requests == []


def test_mem_search_server_error_is_a_json_error(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    fake_engram.route("GET", "/search", (500, {"error": "boom"}))
    provider = make_provider(fake_engram.base_url)

    assert "boom" in call(provider, "mem_search", query="x")["error"]


# -- mem_save -----------------------------------------------------------------


def test_mem_save_request_status_and_state(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(
        provider,
        "mem_save",
        title="Fixed the loader",
        content="**What**: x <private>secret</private>",
        type="bugfix",
        topic_key="loader/fix",
    )

    body = only(fake_engram, "POST", "/observations").json_body
    assert body == {
        "session_id": SESSION,
        "type": "bugfix",
        "title": "Fixed the loader",
        "content": "**What**: x [REDACTED]",
        "project": PROJECT,
        "scope": "project",
        "topic_key": "loader/fix",
        "project_provenance": UNKNOWN_PROVENANCE,
    }
    assert result["id"] == 900
    assert result["status"] == "🧠 saved #900"
    assert result["engram_status"] == "saved"
    assert result["project"] == PROJECT
    state = read_state(tmp_path)
    assert state["last_memory_id"] == 900
    assert state["last_action"] == "saved"
    assert state["project"] == PROJECT


def test_mem_save_defaults_type_and_scope(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    call(provider, "mem_save", title="T", content="C", project="other")

    body = only(fake_engram, "POST", "/observations").json_body
    assert body["type"] == "manual"
    assert body["scope"] == "project"
    assert body["project"] == "other"
    assert "topic_key" not in body


def test_mem_save_passes_judgment_candidates_through(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    candidates = [{"judgment_id": "rel-1", "id": 12, "title": "Old decision"}]
    fake_engram.route(
        "POST",
        "/observations",
        (201, {"id": 901, "status": "saved", "judgment_required": True, "candidates": candidates}),
    )
    provider = make_provider(fake_engram.base_url)

    result = call(provider, "mem_save", title="T", content="C")

    assert result["status"] == "🧠 saved #901"
    assert result["judgment_required"] is True
    assert result["candidates"] == candidates
    assert "mem_judge" in result["next_step"]


def test_mem_save_without_candidates_has_no_next_step(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    assert "next_step" not in call(provider, "mem_save", title="T", content="C")


def test_mem_save_registers_the_session_and_retries_on_session_not_found(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    attempts: list[int] = []

    def observations(_request: RecordedRequest) -> tuple[int, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            return (404, {"error": "session not found"})
        return (201, {"id": 902, "status": "saved"})

    fake_engram.route("POST", "/observations", observations)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_save", title="T", content="C")

    assert result["status"] == "🧠 saved #902"
    assert len(attempts) == 2
    session = only(fake_engram, "POST", "/sessions").json_body
    assert session["id"] == SESSION
    assert session["project"] == PROJECT


def test_mem_save_validates_required_args(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    assert "title" in call(provider, "mem_save", content="C")["error"]
    assert "content" in call(provider, "mem_save", title="T")["error"]
    assert "string" in call(provider, "mem_save", title=5, content="C")["error"]
    assert fake_engram.requests == []


# -- mem_session_summary ------------------------------------------------------


def test_mem_session_summary_request_status_and_state(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_session_summary", content="## Goal\nship T6")

    body = only(fake_engram, "POST", "/observations").json_body
    assert body == {
        "session_id": SESSION,
        "type": "session_summary",
        "title": "Session summary",
        "content": "## Goal\nship T6",
        "project": PROJECT,
        "scope": "project",
        "project_provenance": UNKNOWN_PROVENANCE,
    }
    assert result["status"] == "🧠 saved #900"
    assert read_state(tmp_path)["last_action"] == "session_summary"


# -- mem_update ---------------------------------------------------------------


def test_mem_update_requires_an_explicit_expected_project() -> None:
    # gentle-engram 0.2.0 / Engram v3 (e8c0ce3): the owner is a caller assertion.
    schema = next(s for s in tools_module.get_tool_schemas() if s["name"] == "mem_update")
    params = schema["parameters"]
    assert params["required"] == ["id", "expected_project"]
    assert params["properties"]["expected_project"]["type"] == "string"
    assert "must match" in params["properties"]["expected_project"]["description"]


def test_mem_update_sends_only_given_fields(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(
        provider, "mem_update", id=761, expected_project=PROJECT, title="New", scope="personal"
    )

    request = only(fake_engram, "PATCH", "/observations/761")
    assert request.json_body == {"title": "New", "scope": "personal"}
    assert request.query == {"expected_project": [PROJECT]}
    assert result["id"] == 761
    assert result["status"] == "🧠 updated #761"
    assert result["observation"]["title"] == "New"
    state = read_state(tmp_path)
    assert (state["last_memory_id"], state["last_action"]) == (761, "updated")


def test_mem_update_forwards_the_callers_project_not_the_detected_one(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    call(provider, "mem_update", id=761, expected_project="  other-project ", title="New")

    request = only(fake_engram, "PATCH", "/observations/761")
    assert request.query_one("expected_project") == "other-project"


@pytest.mark.parametrize("expected_project", [None, "", "   "])
def test_mem_update_without_expected_project_is_refused_before_any_request(
    fake_engram: FakeEngram, make_provider: Any, expected_project: Any
) -> None:
    # Never defaulted to the detected project: a guessed owner would defeat the check.
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    args: dict[str, Any] = {"id": 761, "title": "New"}
    if expected_project is not None:
        args["expected_project"] = expected_project
    error = call(provider, "mem_update", **args)["error"]

    assert "'expected_project' is required" in error
    assert "mem_get_observation" in error
    assert fake_engram.requests == []


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (409, "expected_project does not match observation owner"),
        (409, "observation project cannot be reassigned"),
        (400, "expected_project must be a valid non-empty project name"),
    ],
)
def test_mem_update_surfaces_a_refusal_as_a_readable_error(
    fake_engram: FakeEngram,
    make_provider: Any,
    tmp_path: Path,
    status: int,
    message: str,
) -> None:
    wire(fake_engram)
    fake_engram.route("PATCH", "/observations/761", (status, {"error": message}))
    provider = make_provider(fake_engram.base_url)

    error = call(provider, "mem_update", id=761, expected_project="other", title="New")["error"]

    assert f"HTTP {status} {message}" in error
    assert "Engram refused the update of #761" in error
    # A server answered, so "is it running?" would send the model the wrong way.
    assert "engram serve" not in error
    assert not state_path(tmp_path).exists()


class _StubUpdateClient:
    """A client whose update fails with a chosen reason and status, and nothing else."""

    def __init__(self, reason: str, status: int | None) -> None:
        self._reason = reason
        self.last_error_status = status

    def update_observation(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def last_error_text(self) -> str:
        return self._reason


def _stub_env(client: Any) -> tools_module.ToolEnv:
    return tools_module.ToolEnv(
        client=client,
        session_id=SESSION,
        hermes_home="/nonexistent",
        writes_enabled=True,
        agent_context="primary",
        resolve_project=lambda: PROJECT,
        register_session=lambda _project: True,
    )


@pytest.mark.parametrize(
    ("reason", "status", "refused"),
    [
        # The HTTP status decides, never the wording of the reason.
        ("PATCH /observations/761: owner mismatch", 409, True),
        ("PATCH /observations/761: HTTP 404 observation not found", 404, True),
        ("PATCH /observations/761: HTTP 500 database is locked", 500, True),
        ("PATCH /observations/761: unreachable (proxy said: HTTP 502)", None, False),
        ("PATCH /observations/761: timed out after 3s", None, False),
    ],
)
def test_mem_update_classifies_failures_by_http_status(
    reason: str, status: int | None, refused: bool
) -> None:
    raw = tools_module.handle_tool_call(
        _stub_env(_StubUpdateClient(reason, status)),
        "mem_update",
        {"id": 761, "expected_project": PROJECT, "title": "T"},
    )
    error = json.loads(raw)["error"]

    assert reason in error
    if refused:
        assert error.startswith("Engram refused the update of #761: ")
        assert "engram serve" not in error
    else:
        assert error.startswith("Engram update failed: ")
        assert "engram serve" in error


def test_mem_update_keeps_the_hint_when_engram_is_down(make_provider: Any) -> None:
    provider = make_provider(f"http://127.0.0.1:{unused_port()}")

    error = call(provider, "mem_update", id=761, expected_project=PROJECT, title="T")["error"]

    assert "Is `engram serve` running?" in error
    assert "refused the update" not in error


def test_mem_update_accepts_a_numeric_string_id(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    result = call(provider, "mem_update", id="#761", expected_project=PROJECT, content="c")
    assert result["status"] == "🧠 updated #761"


def test_mem_update_rejects_bad_ids_and_empty_patches(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    for bad in (0, -1, True, "abc", None, 1.5):
        error = call(provider, "mem_update", id=bad, expected_project=PROJECT, title="x")["error"]
        assert "id" in error
    assert "at least one" in call(provider, "mem_update", id=761, expected_project=PROJECT)["error"]
    assert fake_engram.requests == []


# -- mem_get_observation ------------------------------------------------------


def test_mem_get_observation_result_status_and_state(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_get_observation", id=761)

    only(fake_engram, "GET", "/observations/761")
    assert result["status"] == "🧠 observation #761"
    assert result["observation"] == OBSERVATION
    state = read_state(tmp_path)
    assert state == {**state, "last_memory_id": 761, "last_action": "observation"}
    assert state["project"] == PROJECT


def test_mem_get_observation_without_history_sends_no_history_params(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_get_observation", id=761)

    assert only(fake_engram, "GET", "/observations/761").query == {}
    assert set(result) == {"id", "status", "observation"}


HISTORY_PAGE = {
    **OBSERVATION,
    "history": [
        {"version": 2, "title": "Old", "content": "z" * 5000, "created_at": "2026-01-01"},
        {"version": 3, "title": "Older", "content": "short", "created_at": "2026-01-02"},
    ],
    "version_count": 60,
    "history_truncated": True,
    "history_from_version": 2,
    "history_cursor": 50,
}


def test_mem_get_observation_with_history_forwards_and_returns_the_page(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    fake_engram.route("GET", "/observations/761", (200, HISTORY_PAGE))
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_get_observation", id=761, include_history=True, history_cursor=50)

    request = only(fake_engram, "GET", "/observations/761")
    assert request.query == {"include_history": ["true"], "history_cursor": ["50"]}
    assert result["observation"] == OBSERVATION
    assert result["version_count"] == 60
    assert result["history_truncated"] is True
    assert result["history_from_version"] == 2
    assert result["history_cursor"] == 50
    first, second = result["history"]
    assert first["version"] == 2
    assert first["content_truncated"] is True
    assert len(first["content"]) <= tools_module.HISTORY_CONTENT_MAX_CHARS + 3
    assert second == HISTORY_PAGE["history"][1]
    state = read_state(tmp_path)
    assert (state["last_memory_id"], state["last_action"]) == (761, "observation")


def test_mem_get_observation_last_history_page_has_no_cursor(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    last_page = {**OBSERVATION, "history": [], "version_count": 1}
    fake_engram.route("GET", "/observations/761", (200, last_page))
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_get_observation", id=761, include_history=True)

    assert only(fake_engram, "GET", "/observations/761").query == {"include_history": ["true"]}
    assert result["history"] == []
    assert result["version_count"] == 1
    assert result["history_truncated"] is False
    assert "history_cursor" not in result


def test_mem_get_observation_history_on_an_older_engram_says_so(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    # An Engram without history ignores the flag and answers the plain row.
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)

    result = call(provider, "mem_get_observation", id=761, include_history=True)

    assert result["observation"] == OBSERVATION
    assert "history" not in result
    assert "Engram >= 3.5.1" in result["history_note"]


@pytest.mark.parametrize("cursor", [-1, "x", 1.5, True])
def test_mem_get_observation_refuses_a_bad_history_cursor_without_a_request(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path, cursor: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    error = call(
        provider, "mem_get_observation", id=761, include_history=True, history_cursor=cursor
    )["error"]

    assert "history_cursor" in error
    assert fake_engram.requests == []
    assert not state_path(tmp_path).exists()


def test_mem_get_observation_caps_huge_content(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    fake_engram.route("GET", "/observations/761", (200, {**OBSERVATION, "content": "y" * 50000}))
    provider = make_provider(fake_engram.base_url)

    observation = call(provider, "mem_get_observation", id=761)["observation"]

    assert observation["content_truncated"] is True
    assert len(observation["content"]) <= tools_module.OBSERVATION_CONTENT_MAX_CHARS + 3


def test_mem_get_observation_not_found_is_an_error_and_no_state(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)

    result = call(provider, "mem_get_observation", id=5)

    assert "404" in result["error"]
    assert read_state(tmp_path) == {}


# -- mem_context and mem_timeline ---------------------------------------------


def test_mem_context_request_and_result(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_context", scope="project")

    request = only(fake_engram, "GET", "/context")
    assert request.query_one("project") == PROJECT
    assert request.query_one("scope") == "project"
    assert result == {"project": PROJECT, "context": "## Recent\n- did a thing"}


def test_mem_timeline_request_and_result(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_timeline", observation_id=761, before=2, after=99)

    request = only(fake_engram, "GET", "/timeline")
    assert request.query_one("observation_id") == "761"
    assert request.query_one("before") == "2"
    assert request.query_one("after") == str(tools_module.TIMELINE_WINDOW_MAX)
    assert request.query_one("project") == PROJECT
    assert result["focus"]["id"] == 761


# -- mem_judge ----------------------------------------------------------------


def test_mem_judge_request_and_result(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(
        provider,
        "mem_judge",
        judgment_id="rel-1",
        relation="supersedes",
        reason="newer decision",
        confidence=0.9,
    )

    assert only(fake_engram, "POST", "/conflicts/judge").json_body == {
        "judgment_id": "rel-1",
        "relation": "supersedes",
        "reason": "newer decision",
        "confidence": 0.9,
        "session_id": SESSION,
    }
    assert result["relation"]["relation"] == "supersedes"
    assert result["status"] == "🧠 judged rel-1: supersedes"


def test_mem_judge_validates_relation_and_confidence(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    assert "relation" in call(provider, "mem_judge", judgment_id="r", relation="maybe")["error"]
    assert (
        "confidence"
        in call(provider, "mem_judge", judgment_id="r", relation="related", confidence=2)["error"]
    )
    assert "judgment_id" in call(provider, "mem_judge", relation="related")["error"]
    assert fake_engram.requests == []


# -- mem_judge_many -----------------------------------------------------------

JUDGE_MANY_OUTCOMES = {
    "outcomes": [
        {
            "index": 0,
            "judgment_id": "rel-1",
            "status": "success",
            "relation": {"judgment_id": "rel-1", "relation": "supersedes"},
            "judgment_status": "judged",
        },
        {
            "index": 1,
            "judgment_id": "rel-2",
            "status": "error",
            "error": 'JudgeRelation: relation "rel-2" not found',
        },
    ]
}


def test_mem_judge_many_request_and_result(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    fake_engram.route("POST", "/conflicts/judge-many", (200, JUDGE_MANY_OUTCOMES))
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(
        provider,
        "mem_judge_many",
        judgments=[
            {"judgment_id": "rel-1", "relation": "supersedes", "reason": "newer", "confidence": 1},
            {"judgment_id": " rel-2 ", "relation": "related", "evidence": "same file"},
        ],
    )

    assert only(fake_engram, "POST", "/conflicts/judge-many").json_body == {
        "judgments": [
            {
                "judgment_id": "rel-1",
                "relation": "supersedes",
                "reason": "newer",
                "confidence": 1.0,
            },
            {"judgment_id": "rel-2", "relation": "related", "evidence": "same file"},
        ],
        "session_id": SESSION,
    }
    assert [(r.method, r.path) for r in fake_engram.requests] == [("POST", "/conflicts/judge-many")]
    assert result["outcomes"] == JUDGE_MANY_OUTCOMES["outcomes"]
    assert result["status"] == "🧠 judged 1 of 2 (1 failed)"
    assert not state_path(tmp_path).exists()


@pytest.mark.parametrize(
    ("judgments", "message"),
    [
        (None, "'judgments'"),
        ([], "1..32"),
        ([{"judgment_id": f"r{i}", "relation": "related"} for i in range(33)], "1..32"),
        ("rel-1", "'judgments'"),
        (["rel-1"], "judgments[0]"),
        ([{"judgment_id": "r", "relation": "related"}, {"relation": "related"}], "judgments[1]"),
        ([{"judgment_id": "r", "relation": "maybe"}], "judgments[0]"),
        ([{"judgment_id": "r", "relation": "related", "confidence": 2}], "judgments[0]"),
        ([{"judgment_id": "r", "relation": "related", "reason": 5}], "judgments[0]"),
    ],
)
def test_mem_judge_many_refuses_a_bad_batch_without_a_request(
    fake_engram: FakeEngram, make_provider: Any, judgments: Any, message: str
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    args = {} if judgments is None else {"judgments": judgments}
    error = call(provider, "mem_judge_many", **args)["error"]

    assert message in error
    assert fake_engram.requests == []


def test_mem_judge_many_is_never_replayed_after_a_server_error(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    fake_engram.route("POST", "/conflicts/judge-many", (500, {"error": "database is locked"}))
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    error = call(
        provider, "mem_judge_many", judgments=[{"judgment_id": "r", "relation": "related"}]
    )["error"]

    assert "database is locked" in error
    assert "engram serve" not in error
    assert [(r.method, r.path) for r in fake_engram.requests] == [("POST", "/conflicts/judge-many")]


@pytest.mark.parametrize(
    "older",
    [Malformed(404, b"404 page not found\n"), Malformed(405, b"Method Not Allowed\n")],
)
def test_mem_judge_many_on_an_older_engram_needs_3_5_1(
    fake_engram: FakeEngram, make_provider: Any, older: Malformed
) -> None:
    wire(fake_engram)
    fake_engram.route("POST", "/conflicts/judge-many", older)
    provider = make_provider(fake_engram.base_url)

    error = call(
        provider, "mem_judge_many", judgments=[{"judgment_id": "r", "relation": "related"}]
    )["error"]

    assert "mem_judge_many needs Engram >= 3.5.1" in error
    assert "engram serve" not in error


# -- mem_consolidate ----------------------------------------------------------

PLAN = {
    "source_ids": [11, 12],
    "sources": [{"id": 11, "title": "A"}, {"id": 12, "title": "B"}],
    "project": PROJECT,
    "fingerprint": "fp-1",
    "survivor": {"title": "Merged", "content": "Both", "project": PROJECT},
}
CONSOLIDATE_ARGS: dict[str, Any] = {
    "ids_to_delete": [11, "#12"],
    "new_title": "Merged <private>t</private>",
    "new_content": "Both <private>secret</private>",
    "expected_project": PROJECT,
}


def consolidations(fake: FakeEngram) -> list[RecordedRequest]:
    return [r for r in fake.requests if r.path == "/observations/consolidate"]


def test_mem_consolidate_preview_returns_the_plan_and_asks_for_approval(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    fake_engram.route(
        "POST", "/observations/consolidate", (200, {"status": "preview", "plan": PLAN})
    )
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_consolidate", **CONSOLIDATE_ARGS, type="decision")

    assert only(fake_engram, "POST", "/observations/consolidate").json_body == {
        "ids_to_delete": [11, 12],
        "new_title": "Merged [REDACTED]",
        "new_content": "Both [REDACTED]",
        "expected_project": PROJECT,
        "type": "decision",
    }
    assert [r.path for r in fake_engram.requests] == ["/observations/consolidate"]
    assert result["engram_status"] == "preview"
    assert result["plan"] == PLAN
    assert result["fingerprint"] == "fp-1"
    assert result["status"] == "🧠 consolidation preview: 2 observations into 1"
    assert "user" in result["next_step"]
    assert "confirm: true" in result["next_step"]
    assert '"fp-1"' in result["next_step"]
    assert not state_path(tmp_path).exists()


def test_mem_consolidate_confirm_applies_and_remembers_the_survivor(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    survivor = {"id": 30, "title": "Merged", "project": PROJECT, "type": "decision"}
    fake_engram.route(
        "POST",
        "/observations/consolidate",
        (
            201,
            {
                "status": "consolidated",
                "plan": PLAN,
                "survivor": survivor,
                "soft_deleted_ids": [11, 12],
            },
        ),
    )
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(
        provider,
        "mem_consolidate",
        **CONSOLIDATE_ARGS,
        topic_key="merge/ab",
        confirm=True,
        fingerprint="fp-1",
    )

    assert only(fake_engram, "POST", "/observations/consolidate").json_body == {
        "ids_to_delete": [11, 12],
        "new_title": "Merged [REDACTED]",
        "new_content": "Both [REDACTED]",
        "expected_project": PROJECT,
        "topic_key": "merge/ab",
        "confirm": True,
        "fingerprint": "fp-1",
    }
    assert result["id"] == 30
    assert result["status"] == "🧠 consolidated #30 (soft-deleted #11, #12)"
    assert result["engram_status"] == "consolidated"
    assert result["soft_deleted_ids"] == [11, 12]
    assert result["survivor"] == survivor
    state = read_state(tmp_path)
    assert (state["last_memory_id"], state["last_action"], state["project"]) == (
        30,
        "consolidated",
        PROJECT,
    )


def test_mem_consolidate_fingerprint_mismatch_returns_the_fresh_plan_unconfirmed(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    fresh = {**PLAN, "fingerprint": "fp-2"}
    fake_engram.route(
        "POST",
        "/observations/consolidate",
        (
            409,
            {
                "error": "consolidation fingerprint mismatch",
                "code": "fingerprint_mismatch",
                "plan": fresh,
            },
        ),
    )
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_consolidate", **CONSOLIDATE_ARGS, confirm=True, fingerprint="fp-1")

    # Never auto-confirmed with the fresh fingerprint: exactly one request.
    assert len(consolidations(fake_engram)) == 1
    assert result["code"] == "fingerprint_mismatch"
    assert result["plan"] == fresh
    assert "Nothing was consolidated" in result["error"]
    assert "user" in result["error"]
    assert not state_path(tmp_path).exists()


def test_mem_consolidate_is_never_retried_after_a_server_failure(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    fake_engram.route(
        "POST",
        "/observations/consolidate",
        (500, {"error": "consolidation failed", "code": "consolidation_failed"}),
    )
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_consolidate", **CONSOLIDATE_ARGS, confirm=True, fingerprint="fp-1")

    assert len(consolidations(fake_engram)) == 1
    assert result["code"] == "consolidation_failed"
    assert "outcome is unknown" in result["error"]
    assert "do not retry" in result["error"]
    assert not state_path(tmp_path).exists()


def test_mem_consolidate_is_never_retried_after_a_timeout(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    from hermes_recall import client as client_module

    wire(fake_engram)
    fake_engram.route(
        "POST", "/observations/consolidate", Delay(1.0, (201, {"status": "consolidated"}))
    )
    provider = make_provider(fake_engram.base_url)
    monkeypatch.setattr(client_module, "WRITE_TIMEOUT_S", 0.2)
    fake_engram.reset()

    result = call(provider, "mem_consolidate", **CONSOLIDATE_ARGS, confirm=True, fingerprint="fp-1")

    assert len(consolidations(fake_engram)) == 1
    assert "outcome is unknown" in result["error"]
    assert "engram serve" not in result["error"]
    assert not state_path(tmp_path).exists()


def test_mem_consolidate_reports_an_owner_refusal_with_its_code(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    fake_engram.route(
        "POST",
        "/observations/consolidate",
        (403, {"error": "observation project mismatch", "code": "project_mismatch"}),
    )
    provider = make_provider(fake_engram.base_url)

    result = call(provider, "mem_consolidate", **CONSOLIDATE_ARGS)

    assert result == {
        "error": "Engram refused the consolidation: POST /observations/consolidate: "
        "HTTP 403 observation project mismatch",
        "code": "project_mismatch",
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"ids_to_delete": None}, "ids_to_delete"),
        ({"ids_to_delete": [11]}, "ids_to_delete"),
        ({"ids_to_delete": [11, 11]}, "distinct"),
        ({"ids_to_delete": [11, "#11"]}, "distinct"),
        ({"ids_to_delete": [11, 0]}, "ids_to_delete"),
        ({"ids_to_delete": [11, -2]}, "ids_to_delete"),
        ({"ids_to_delete": [11, True]}, "ids_to_delete"),
        ({"ids_to_delete": "11,12"}, "ids_to_delete"),
        ({"new_title": " "}, "new_title"),
        ({"new_content": "  "}, "new_content"),
        ({"expected_project": ""}, "expected_project"),
        ({"confirm": True}, "fingerprint"),
    ],
)
def test_mem_consolidate_refuses_bad_arguments_without_a_request(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path, override: Any, message: str
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    args = {**CONSOLIDATE_ARGS, **override}
    args = {key: value for key, value in args.items() if value is not None}
    error = call(provider, "mem_consolidate", **args)["error"]

    assert message in error
    assert fake_engram.requests == []
    assert not state_path(tmp_path).exists()


@pytest.mark.parametrize(
    "older",
    [Malformed(404, b"404 page not found\n"), Malformed(405, b"Method Not Allowed\n")],
)
def test_mem_consolidate_on_an_older_engram_needs_3_5_1(
    fake_engram: FakeEngram, make_provider: Any, older: Malformed
) -> None:
    wire(fake_engram)
    fake_engram.route("POST", "/observations/consolidate", older)
    provider = make_provider(fake_engram.base_url)

    error = call(provider, "mem_consolidate", **CONSOLIDATE_ARGS)["error"]

    assert "mem_consolidate needs Engram >= 3.5.1" in error


# -- mem_session_finalize -----------------------------------------------------

FINALIZE_PATH = f"/sessions/{SESSION}/finalize"
FINALIZED = {
    "session_id": SESSION,
    "state": "finalized",
    "summary_id": 950,
    "summary_sync_id": "obs-950",
    "already_finalized": False,
    "judgment_required": False,
}


def paths(fake: FakeEngram) -> list[tuple[str, str]]:
    return [(r.method, r.path) for r in fake.requests]


def end_paths(fake: FakeEngram) -> list[str]:
    return [r.path for r in fake.requests if r.path.endswith("/end")]


def finish(provider: RecallProvider) -> None:
    """The host's shutdown: end the session, then drain the worker."""
    provider.on_session_end([])
    assert provider._drain_for_test()


def test_mem_session_finalize_request_result_state_and_no_end(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    fake_engram.route("POST", FINALIZE_PATH, (200, FINALIZED))
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_session_finalize", summary="## Goal\nship <private>key</private>")

    # Registered at initialize, so nothing but the finalize itself; the body is
    # exactly {project, summary}: Engram refuses any other key.
    assert paths(fake_engram) == [("POST", FINALIZE_PATH)]
    assert only(fake_engram, "POST", FINALIZE_PATH).json_body == {
        "project": PROJECT,
        "summary": "## Goal\nship [REDACTED]",
    }
    assert result["id"] == 950
    assert result["state"] == "finalized"
    assert result["status"] == "🧠 session finalized, summary #950"
    state = read_state(tmp_path)
    assert (state["last_memory_id"], state["last_action"]) == (950, "session_summary")

    finish(provider)
    provider.on_session_switch("sess-2", reset=True)
    assert provider._drain_for_test()
    assert end_paths(fake_engram) == []


def test_mem_session_finalize_already_finalized_also_suppresses_the_end(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    again = {**FINALIZED, "state": "already_finalized", "already_finalized": True}
    fake_engram.route("POST", FINALIZE_PATH, (200, again))
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_session_finalize", summary="done")

    assert result["status"] == "🧠 session already finalized, summary #950"
    finish(provider)
    assert end_paths(fake_engram) == []


def test_mem_session_finalize_registers_the_session_first(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    fake_engram.route("POST", "/sessions", (500, {"error": "database is locked"}))
    fake_engram.route("POST", FINALIZE_PATH, (200, FINALIZED))
    provider = make_provider(fake_engram.base_url)
    fake_engram.route("POST", "/sessions", echo_session)
    fake_engram.reset()

    call(provider, "mem_session_finalize", summary="done")

    assert paths(fake_engram) == [("POST", "/sessions"), ("POST", FINALIZE_PATH)]


def test_mem_session_finalize_is_not_sent_when_registration_fails(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    fake_engram.route("POST", "/sessions", (500, {"error": "database is locked"}))
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    error = call(provider, "mem_session_finalize", summary="done")["error"]

    assert "database is locked" in error
    assert paths(fake_engram) == [("POST", "/sessions")]


def test_a_write_after_finalize_registers_a_continuation_and_its_end_is_posted(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    fake_engram.route("POST", FINALIZE_PATH, (200, FINALIZED))
    fake_engram.route("POST", "/prompts", (201, {"id": 1, "status": "saved"}))
    provider = make_provider(fake_engram.base_url)
    continuation = f"{SESSION}:resume:2"
    fake_engram.route("POST", "/sessions", (201, {"id": continuation, "status": "created"}))
    call(provider, "mem_session_finalize", summary="done")
    fake_engram.reset()

    provider.on_turn_start(2, "one more question after the summary")
    finish(provider)

    assert only(fake_engram, "POST", "/sessions").json_body["resume"] is True
    assert only(fake_engram, "POST", "/prompts").json_body["session_id"] == continuation
    # The continuation is live, so its end is posted (the id is path-quoted).
    assert end_paths(fake_engram) == [f"/sessions/{SESSION}%3Aresume%3A2/end"]


def test_a_tool_save_after_finalize_registers_a_continuation_first(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    # Seen against a real Engram 3.6.0: without this, mem_save wrote straight into
    # the finalized root session while background captures used a continuation.
    wire(fake_engram)
    fake_engram.route("POST", FINALIZE_PATH, (200, FINALIZED))
    provider = make_provider(fake_engram.base_url)
    continuation = f"{SESSION}:resume:2"
    fake_engram.route("POST", "/sessions", (201, {"id": continuation, "status": "created"}))
    call(provider, "mem_session_finalize", summary="done")
    fake_engram.reset()

    result = call(provider, "mem_save", title="After", content="one more fact")

    assert paths(fake_engram) == [("POST", "/sessions"), ("POST", "/observations")]
    assert only(fake_engram, "POST", "/sessions").json_body["resume"] is True
    assert only(fake_engram, "POST", "/observations").json_body["session_id"] == continuation
    assert result["id"] == 900


def test_a_tool_save_in_a_registered_session_sends_no_registration(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    call(provider, "mem_save", title="T", content="C")

    assert paths(fake_engram) == [("POST", "/observations")]


@pytest.mark.parametrize(
    "answer",
    [
        (500, {"error": "session finalization failed", "code": "session_finalize_failed"}),
        (200, {**FINALIZED, "state": "pending"}),
        (200, {**FINALIZED, "session_id": "someone-else"}),
        Delay(1.0, (200, FINALIZED)),
    ],
)
def test_an_uncertain_finalize_is_never_retried_nor_followed_by_an_end(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path, monkeypatch: Any, answer: Any
) -> None:
    from hermes_recall import client as client_module

    wire(fake_engram)
    fake_engram.route("POST", FINALIZE_PATH, answer)
    provider = make_provider(fake_engram.base_url)
    monkeypatch.setattr(client_module, "WRITE_TIMEOUT_S", 0.2)
    fake_engram.reset()

    error = call(provider, "mem_session_finalize", summary="done")["error"]

    assert "outcome is unknown" in error
    assert "do not retry" in error
    assert paths(fake_engram) == [("POST", FINALIZE_PATH)]
    assert not state_path(tmp_path).exists()
    finish(provider)
    assert end_paths(fake_engram) == []


@pytest.mark.parametrize(
    ("status", "code", "end_posted"),
    [
        (403, "session_project_mismatch", True),
        (404, "unknown_session", True),
        (400, "invalid_arguments", True),
        # Already ended by someone else: another /end would be redundant.
        (409, "session_already_ended", False),
    ],
)
def test_a_refused_finalize_reports_engrams_code(
    fake_engram: FakeEngram,
    make_provider: Any,
    tmp_path: Path,
    status: int,
    code: str,
    end_posted: bool,
) -> None:
    wire(fake_engram)
    fake_engram.route(
        "POST", FINALIZE_PATH, (status, {"error": "refused", "code": code, "session_id": SESSION})
    )
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_session_finalize", summary="done", project="other")

    assert only(fake_engram, "POST", FINALIZE_PATH).json_body == {
        "project": "other",
        "summary": "done",
    }
    assert result["code"] == code
    assert f"HTTP {status} refused" in result["error"]
    assert "engram serve" not in result["error"]
    assert paths(fake_engram) == [("POST", FINALIZE_PATH)]
    assert not state_path(tmp_path).exists()
    finish(provider)
    assert end_paths(fake_engram) == ([f"/sessions/{SESSION}/end"] if end_posted else [])


@pytest.mark.parametrize(
    "older",
    [Malformed(404, b"404 page not found\n"), Malformed(405, b"Method Not Allowed\n")],
)
def test_mem_session_finalize_on_an_older_engram_needs_3_5_1_and_keeps_the_end(
    fake_engram: FakeEngram, make_provider: Any, older: Malformed
) -> None:
    wire(fake_engram)
    fake_engram.route("POST", FINALIZE_PATH, older)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    error = call(provider, "mem_session_finalize", summary="done")["error"]

    assert "mem_session_finalize needs Engram >= 3.5.1" in error
    finish(provider)
    assert end_paths(fake_engram) == [f"/sessions/{SESSION}/end"]


def test_mem_session_finalize_requires_a_summary(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    assert "summary" in call(provider, "mem_session_finalize")["error"]
    assert "summary" in call(provider, "mem_session_finalize", summary="  ")["error"]
    assert fake_engram.requests == []


# -- server refusals are not outages (S8) --------------------------------------


def test_a_read_refused_with_unknown_project_reports_engrams_reason(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    refusal = {
        "error": 'project "nope" not found',
        "code": "unknown_project",
        "available_projects": ["hermes-recall", "engram"],
    }
    fake_engram.route("GET", "/search", (404, refusal))
    provider = make_provider(fake_engram.base_url)

    result = call(provider, "mem_search", query="x", project="nope")

    assert result["error"] == (
        'Engram refused the search: GET /search: HTTP 404 project "nope" not found'
    )
    assert result["code"] == "unknown_project"
    assert result["available_projects"] == ["hermes-recall", "engram"]


def test_a_read_refused_with_ambiguous_project_reports_engrams_reason(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    refusal = {
        "error": "ambiguous project",
        "code": "ambiguous_project",
        "available_projects": ["a", "b"],
        "project_source": "ambiguous",
    }
    fake_engram.route("GET", "/context", (409, refusal))
    provider = make_provider(fake_engram.base_url)

    result = call(provider, "mem_context")

    assert result == {
        "error": "Engram refused the context: GET /context: HTTP 409 ambiguous project",
        "code": "ambiguous_project",
        "available_projects": ["a", "b"],
    }


# -- mem_pin / mem_unpin ------------------------------------------------------


def test_mem_pin_and_unpin(fake_engram: FakeEngram, make_provider: Any, tmp_path: Path) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    pinned = call(provider, "mem_pin", id=761)
    only(fake_engram, "PUT", "/observations/761/pin")
    assert pinned == {"id": 761, "pinned": True, "status": "🧠 pinned #761"}
    assert read_state(tmp_path)["last_action"] == "pinned"

    unpinned = call(provider, "mem_unpin", id=761)
    only(fake_engram, "DELETE", "/observations/761/pin")
    assert unpinned == {"id": 761, "pinned": False, "status": "🧠 unpinned #761"}
    assert read_state(tmp_path)["last_action"] == "unpinned"


def test_mem_pin_missing_observation_is_an_error(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    fake_engram.route("PUT", "/observations/5/pin", (404, {"error": "observation not found"}))
    provider = make_provider(fake_engram.base_url)

    assert "observation not found" in call(provider, "mem_pin", id=5)["error"]


# -- non-primary contexts -----------------------------------------------------


@pytest.mark.parametrize("agent_context", ["subagent", "cron", "flush"])
@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("mem_save", {"title": "T", "content": "C"}),
        ("mem_update", {"id": 761, "expected_project": "p", "title": "T"}),
        ("mem_session_summary", {"content": "C"}),
        ("mem_judge", {"judgment_id": "r", "relation": "related"}),
        ("mem_pin", {"id": 761}),
        ("mem_unpin", {"id": 761}),
        ("mem_judge_many", {"judgments": [{"judgment_id": "r", "relation": "related"}]}),
        ("mem_consolidate", CONSOLIDATE_ARGS),
        ("mem_session_finalize", {"summary": "done"}),
    ],
)
def test_non_primary_write_tools_are_refused(
    fake_engram: FakeEngram,
    make_provider: Any,
    tmp_path: Path,
    agent_context: str,
    name: str,
    args: dict[str, Any],
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url, agent_context=agent_context)
    fake_engram.reset()

    result = json.loads(provider.handle_tool_call(name, args))

    assert "disabled for subagents" in result["error"]
    assert agent_context in result["error"]
    assert fake_engram.requests == []
    assert not state_path(tmp_path).exists()


def test_non_primary_read_tools_still_work(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url, agent_context="subagent")

    assert call(provider, "mem_search", query="x")["count"] == 1
    assert call(provider, "mem_get_observation", id=761)["id"] == 761


# -- fail-soft ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("mem_search", {"query": "x"}),
        ("mem_save", {"title": "T", "content": "C", "project": "p"}),
        ("mem_update", {"id": 761, "expected_project": "p", "title": "T"}),
        ("mem_get_observation", {"id": 761}),
        ("mem_context", {"project": "p"}),
        ("mem_session_summary", {"content": "C", "project": "p"}),
        ("mem_timeline", {"observation_id": 761, "project": "p"}),
        ("mem_judge", {"judgment_id": "r", "relation": "related"}),
        ("mem_pin", {"id": 761}),
        ("mem_unpin", {"id": 761}),
        ("mem_judge_many", {"judgments": [{"judgment_id": "r", "relation": "related"}]}),
        ("mem_consolidate", CONSOLIDATE_ARGS),
        ("mem_session_finalize", {"summary": "done", "project": "p"}),
    ],
)
def test_engram_down_is_a_json_error_never_a_raise(
    make_provider: Any, tmp_path: Path, name: str, args: dict[str, Any]
) -> None:
    provider = make_provider(f"http://127.0.0.1:{unused_port()}")

    result = json.loads(provider.handle_tool_call(name, args))

    assert "engram serve" in result["error"]
    assert not state_path(tmp_path).exists()


def test_unknown_tool_and_bad_args_are_json_errors(make_provider: Any) -> None:
    provider = make_provider(f"http://127.0.0.1:{unused_port()}")

    assert "unknown tool" in json.loads(provider.handle_tool_call("mem_delete", {}))["error"]
    assert "JSON object" in json.loads(provider.handle_tool_call("mem_search", [1]))["error"]
    assert "JSON object" in json.loads(provider.handle_tool_call("mem_search", "{nope"))["error"]


def test_json_string_arguments_are_accepted(fake_engram: FakeEngram, make_provider: Any) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    raw = provider.handle_tool_call("mem_get_observation", '{"id": 761}')
    assert json.loads(raw)["id"] == 761


def test_disabled_provider_answers_every_tool_with_an_error() -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1", enabled=False))
    assert (
        "disabled" in json.loads(provider.handle_tool_call("mem_search", {"query": "x"}))["error"]
    )


def test_a_handler_bug_still_returns_json(make_provider: Any, monkeypatch: Any) -> None:
    provider = make_provider(f"http://127.0.0.1:{unused_port()}")

    def explode(_env: Any, _args: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setitem(tools_module._HANDLERS, "mem_search", explode)

    result = json.loads(provider.handle_tool_call("mem_search", {"query": "x"}))
    assert result == {"error": "mem_search failed: RuntimeError"}


def test_state_write_failure_is_swallowed(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    # A directory where state.json must go makes every state write fail. (The
    # recall/ directory itself already exists: initialize records the server
    # lifecycle outcome in recall/server.json.)
    (tmp_path / "recall" / "state.json").mkdir(parents=True, exist_ok=True)

    result = call(provider, "mem_save", title="T", content="C")

    assert result["status"] == "🧠 saved #900"


def test_memory_tool_results_are_never_passively_captured() -> None:
    assert provider_module._MEMORY_TOOL_NAMES == tools_module.TOOL_NAMES


# -- state.json is written from the primary context only -----------------------


@pytest.mark.parametrize("agent_context", ["subagent", "cron", "flush"])
def test_non_primary_read_tools_do_not_write_state(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path, agent_context: str
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url, agent_context=agent_context)

    assert call(provider, "mem_get_observation", id=761)["id"] == 761

    assert not state_path(tmp_path).exists()
