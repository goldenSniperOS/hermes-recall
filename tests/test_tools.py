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
from fake_engram import FakeEngram, RecordedRequest

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


def wire(fake: FakeEngram) -> None:
    fake.route("GET", "/project/current", (200, {"project": PROJECT}))
    fake.route("POST", "/sessions", (201, {"id": SESSION, "status": "created"}))
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


def test_exactly_ten_schemas_with_the_expected_names() -> None:
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
    assert len(json.dumps(tools_module.get_tool_schemas())) < 4500


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
    }
    assert result["status"] == "🧠 saved #900"
    assert read_state(tmp_path)["last_action"] == "session_summary"


# -- mem_update ---------------------------------------------------------------


def test_mem_update_sends_only_given_fields(
    fake_engram: FakeEngram, make_provider: Any, tmp_path: Path
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    result = call(provider, "mem_update", id=761, title="New", scope="personal")

    assert only(fake_engram, "PATCH", "/observations/761").json_body == {
        "title": "New",
        "scope": "personal",
    }
    assert result["id"] == 761
    assert result["status"] == "🧠 updated #761"
    assert result["observation"]["title"] == "New"
    state = read_state(tmp_path)
    assert (state["last_memory_id"], state["last_action"]) == (761, "updated")


def test_mem_update_accepts_a_numeric_string_id(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    assert call(provider, "mem_update", id="#761", content="c")["status"] == "🧠 updated #761"


def test_mem_update_rejects_bad_ids_and_empty_patches(
    fake_engram: FakeEngram, make_provider: Any
) -> None:
    wire(fake_engram)
    provider = make_provider(fake_engram.base_url)
    fake_engram.reset()

    for bad in (0, -1, True, "abc", None, 1.5):
        assert "id" in call(provider, "mem_update", id=bad, title="x")["error"]
    assert "at least one" in call(provider, "mem_update", id=761)["error"]
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
        ("mem_update", {"id": 761, "title": "T"}),
        ("mem_session_summary", {"content": "C"}),
        ("mem_judge", {"judgment_id": "r", "relation": "related"}),
        ("mem_pin", {"id": 761}),
        ("mem_unpin", {"id": 761}),
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
        ("mem_update", {"id": 761, "title": "T"}),
        ("mem_get_observation", {"id": 761}),
        ("mem_context", {"project": "p"}),
        ("mem_session_summary", {"content": "C", "project": "p"}),
        ("mem_timeline", {"observation_id": 761, "project": "p"}),
        ("mem_judge", {"judgment_id": "r", "relation": "related"}),
        ("mem_pin", {"id": 761}),
        ("mem_unpin", {"id": 761}),
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
    # A regular file where the recall/ directory must go makes every write fail.
    (tmp_path / "recall").write_text("not a directory", encoding="utf-8")

    result = call(provider, "mem_save", title="T", content="C")

    assert result["status"] == "🧠 saved #900"


def test_memory_tool_results_are_never_passively_captured() -> None:
    from hermes_recall import provider as provider_module

    assert provider_module._MEMORY_TOOL_NAMES == tools_module.TOOL_NAMES
