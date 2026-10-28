"""The proxy's bindings: session/project injection (S11), the write gate (S12), side effects (S13).

The ``engram`` binary is :mod:`fake_mcp`, which records the exact arguments every
``tools/call`` received; the HTTP side (session registration, ``/end``) is a
:class:`~fake_engram.FakeEngram` on the port ``ENGRAM_PORT`` names. Never the real
binary, never 127.0.0.1:7437, never ``~/.engram``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_engram import FakeEngram, RecordedRequest
from fake_mcp import AGENT_TOOLS, FakeMcp

from hermes_recall.config import RecallConfig
from hermes_recall.provider import RecallProvider
from hermes_recall.state import read_state

PROJECT = "hermes-recall"
SESSION = "sess-1"

READ_TOOLS = [
    "mem_search",
    "mem_search_prompts",
    "mem_context",
    "mem_get_observation",
    "mem_current_project",
    "mem_list_projects",
    "mem_suggest_topic_key",
    "mem_doctor",
]
WRITE_TOOLS = [tool["name"] for tool in AGENT_TOOLS if tool["name"] not in READ_TOOLS]


def make_config(url: str, **over: Any) -> RecallConfig:
    values: dict[str, Any] = {
        "url": url,
        "project_override": "",
        "recall_limit": 5,
        "recall_max_chars": 2500,
        "enabled": True,
        "auth_token": "",
        "autostart": False,
    }
    values.update(over)
    return RecallConfig(**values)


def text_result(payload: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": json.dumps(payload)}]}
    if is_error:
        result["isError"] = True
    return result


def result_of(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def args_of(fake: FakeMcp, tool: str) -> list[dict[str, Any]]:
    return [arguments for name, arguments in fake.calls() if name == tool]


def session_posts(fake: FakeEngram) -> list[RecordedRequest]:
    return [r for r in fake.requests if r.method == "POST" and r.path == "/sessions"]


def end_posts(fake: FakeEngram) -> list[RecordedRequest]:
    return [
        r
        for r in fake.requests
        if r.method == "POST" and r.path.startswith("/sessions/") and r.path.endswith("/end")
    ]


class Acks:
    """``POST /sessions`` answers: the runtime id first, then resume continuations."""

    def __init__(self) -> None:
        self.ids: list[str] = []

    def __call__(self, request: RecordedRequest) -> tuple[int, Any]:
        runtime = request.json_body["id"]
        effective = runtime if not self.ids else f"{runtime}:resume:{len(self.ids) + 1}"
        self.ids.append(effective)
        return (201, {"id": effective, "status": "created"})


@pytest.fixture
def engram(fake_engram: FakeEngram, monkeypatch: pytest.MonkeyPatch) -> FakeEngram:
    monkeypatch.setenv("ENGRAM_PORT", str(fake_engram.port))
    fake_engram.route("GET", "/project/current", (200, {"project": PROJECT}))
    fake_engram.route("POST", "/sessions", Acks())
    fake_engram.route("POST", f"/sessions/{SESSION}/end", (200, {"status": "ended"}))
    return fake_engram


@pytest.fixture
def make_provider(hermes_home: Path, engram: FakeEngram, fake_mcp: FakeMcp) -> Iterator[Any]:
    built: list[RecallProvider] = []

    def factory(*, agent_context: str = "primary", **over: Any) -> RecallProvider:
        provider = RecallProvider(config=make_config(engram.base_url, **over))
        provider.get_tool_schemas()
        provider.initialize(
            SESSION, hermes_home=str(hermes_home), platform="cli", agent_context=agent_context
        )
        assert provider._drain_for_test()
        built.append(provider)
        return provider

    yield factory
    for provider in built:
        provider.shutdown()


# -- S11 session and project injection ------------------------------------------------


def test_write_gets_the_registered_session_and_detected_project(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    provider = make_provider()

    provider.handle_tool_call(
        "mem_save", {"title": "t", "content": "c", "session_id": "invented-by-the-model"}
    )

    assert args_of(fake_mcp, "mem_save") == [
        {"title": "t", "content": "c", "session_id": SESSION, "project": PROJECT}
    ]
    assert [r.json_body["id"] for r in session_posts(engram)] == [SESSION]


def test_injected_session_is_the_effective_resume_id(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    engram.route("POST", "/sessions", (201, {"id": f"{SESSION}:resume:3", "status": "created"}))
    provider = make_provider()

    provider.handle_tool_call("mem_session_summary", {"content": "Goal: x"})

    assert args_of(fake_mcp, "mem_session_summary") == [
        {"content": "Goal: x", "session_id": f"{SESSION}:resume:3", "project": PROJECT}
    ]


def test_unregistered_session_is_registered_before_the_write(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    engram.route("POST", "/sessions", (500, {"error": "down"}))
    provider = make_provider()  # initialize's registration fails
    engram.route("POST", "/sessions", Acks())
    engram.reset()

    provider.handle_tool_call("mem_save_prompt", {"content": "what the user asked"})

    assert [r.json_body["id"] for r in session_posts(engram)] == [SESSION]
    assert args_of(fake_mcp, "mem_save_prompt")[0]["session_id"] == SESSION


def test_failed_registration_forwards_without_a_session(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    engram.route("POST", "/sessions", (500, {"error": "down"}))
    provider = make_provider()

    provider.handle_tool_call("mem_save", {"title": "t", "content": "c", "session_id": "x"})

    # An id Engram does not know would be refused (unknown_session); without one the
    # write still lands in the project's own manual-save session.
    assert args_of(fake_mcp, "mem_save") == [{"title": "t", "content": "c", "project": PROJECT}]


def test_another_project_on_a_write_drops_the_session(
    make_provider: Any, fake_mcp: FakeMcp
) -> None:
    provider = make_provider()

    provider.handle_tool_call(
        "mem_save", {"title": "t", "content": "c", "project": "other", "session_id": "x"}
    )
    provider.handle_tool_call(
        "mem_save", {"title": "t", "content": "c", "project": "Hermes-Recall"}
    )

    first, second = args_of(fake_mcp, "mem_save")
    assert first == {"title": "t", "content": "c", "project": "other"}
    # Engram canonicalizes names (case, NFC, repeated separators): same project.
    assert second == {
        "title": "t",
        "content": "c",
        "project": "Hermes-Recall",
        "session_id": SESSION,
    }


def test_session_only_tools_get_the_session_and_no_project(
    make_provider: Any, fake_mcp: FakeMcp
) -> None:
    provider = make_provider()

    provider.handle_tool_call("mem_judge", {"judgment_id": "j1", "relation": "related"})
    provider.handle_tool_call("mem_capture_passive", {"content": "## Key Learnings:\n1. x"})

    assert args_of(fake_mcp, "mem_judge") == [
        {"judgment_id": "j1", "relation": "related", "session_id": SESSION}
    ]
    assert args_of(fake_mcp, "mem_capture_passive") == [
        {"content": "## Key Learnings:\n1. x", "session_id": SESSION}
    ]


def test_reads_get_the_project_unless_given_or_searching_everything(
    make_provider: Any, fake_mcp: FakeMcp
) -> None:
    provider = make_provider()

    provider.handle_tool_call("mem_search", {"query": "a"})
    provider.handle_tool_call("mem_search", {"query": "b", "project": "other"})
    provider.handle_tool_call("mem_search", {"query": "c", "all_projects": True})
    provider.handle_tool_call("mem_search_prompts", {"query": "d", "all_projects": "true"})

    assert args_of(fake_mcp, "mem_search") == [
        {"query": "a", "project": PROJECT},
        {"query": "b", "project": "other"},
        {"query": "c", "all_projects": True},
    ]
    assert args_of(fake_mcp, "mem_search_prompts") == [{"query": "d", "all_projects": "true"}]


def test_current_project_gets_the_working_directory(
    make_provider: Any, fake_mcp: FakeMcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    provider = make_provider()

    provider.handle_tool_call("mem_current_project", {})
    provider.handle_tool_call("mem_current_project", {"directory": "/elsewhere"})

    assert args_of(fake_mcp, "mem_current_project") == [
        {"cwd": str(tmp_path)},
        {"directory": "/elsewhere"},
    ]


def test_expected_project_is_never_injected(make_provider: Any, fake_mcp: FakeMcp) -> None:
    provider = make_provider()

    provider.handle_tool_call("mem_update", {"id": 7, "title": "new"})
    provider.handle_tool_call("mem_consolidate", {"ids_to_delete": [1, 2]})

    assert args_of(fake_mcp, "mem_update") == [{"id": 7, "title": "new"}]
    assert args_of(fake_mcp, "mem_consolidate") == [{"ids_to_delete": [1, 2]}]


def test_unknown_session_is_registered_again_and_retried_once(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    provider = make_provider()
    refusal = text_result(
        {"error_code": "unknown_session", "message": "Session was provided but does not exist"},
        is_error=True,
    )
    fake_mcp.configure(results={"mem_save": refusal})
    engram.reset()

    result = result_of(provider.handle_tool_call("mem_save", {"title": "t", "content": "c"}))

    # Registered again, retried once with the fresh effective id, then reported.
    assert [r.json_body["id"] for r in session_posts(engram)] == [SESSION]
    saves = args_of(fake_mcp, "mem_save")
    assert [args["session_id"] for args in saves] == [SESSION, f"{SESSION}:resume:2"]
    assert result["error_code"] == "unknown_session"


def test_session_start_binds_to_recall_session_without_the_child(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    provider = make_provider()
    provider._registered_session_id = ""  # as after an end or a finalize
    engram.reset()

    result = result_of(provider.handle_tool_call("mem_session_start", {"id": "model-made"}))

    assert result["session_id"] == f"{SESSION}:resume:2"
    assert result["project"] == PROJECT
    assert [r.json_body["id"] for r in session_posts(engram)] == [SESSION]
    assert fake_mcp.calls() == []


def test_session_end_ends_recall_session_and_suppresses_the_hook_end(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    provider = make_provider()

    provider.handle_tool_call("mem_session_end", {"id": "model-made", "summary": "done"})
    provider.on_session_end([])
    assert provider._drain_for_test()
    provider.handle_tool_call("mem_save", {"title": "t", "content": "c"})

    assert args_of(fake_mcp, "mem_session_end") == [{"id": SESSION, "summary": "done"}]
    assert end_posts(engram) == []
    # The next write registers again and lands in the continuation.
    assert args_of(fake_mcp, "mem_save")[0]["session_id"] == f"{SESSION}:resume:2"


# -- S12 write gate -----------------------------------------------------------------------


@pytest.mark.parametrize("context", ["subagent", "cron", "flush"])
def test_non_primary_writes_are_refused_without_the_child(
    make_provider: Any, fake_mcp: FakeMcp, context: str
) -> None:
    provider = make_provider(agent_context=context)

    for tool in WRITE_TOOLS:
        result = result_of(provider.handle_tool_call(tool, {"id": 1}))

        assert "writes are disabled" in result["error"], tool
        assert f"agent_context={context}" in result["error"], tool
    assert fake_mcp.calls() == []
    assert len(fake_mcp.starts()) == 1  # discovery only: no call child


def test_a_future_tool_counts_as_a_write(make_provider: Any, fake_mcp: FakeMcp) -> None:
    future = {
        "name": "mem_rewrite_everything",
        "description": "new upstream tool",
        "inputSchema": {"type": "object", "properties": {}},
    }
    fake_mcp.configure(tools=[*AGENT_TOOLS, future])
    provider = make_provider(agent_context="cron")
    assert "mem_rewrite_everything" in [s["name"] for s in provider.get_tool_schemas()]

    result = result_of(provider.handle_tool_call("mem_rewrite_everything", {}))

    assert "writes are disabled" in result["error"]
    assert fake_mcp.calls() == []


def test_non_primary_reads_are_forwarded(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    provider = make_provider(agent_context="subagent")

    for tool in READ_TOOLS:
        assert result_of(provider.handle_tool_call(tool, {"id": 3}))["tool"] == tool

    assert [name for name, _ in fake_mcp.calls()] == READ_TOOLS
    # A guest never registers a session.
    assert session_posts(engram) == []


def test_non_primary_read_never_moves_the_last_memory_id(
    make_provider: Any, fake_mcp: FakeMcp, hermes_home: Path
) -> None:
    provider = make_provider(agent_context="cron")
    fake_mcp.configure(results={"mem_get_observation": text_result({"result": "#3 ..."})})

    result = result_of(provider.handle_tool_call("mem_get_observation", {"id": 3}))

    assert result["status"] == "🧠 observation #3"
    assert read_state(str(hermes_home)) == {}


# -- S13 side effects ---------------------------------------------------------------------


def test_save_result_gets_the_status_line_and_the_last_id(
    make_provider: Any, fake_mcp: FakeMcp, hermes_home: Path
) -> None:
    provider = make_provider()
    saved = {"id": 42, "project": PROJECT, "result": "Memory saved", "state": "active"}
    fake_mcp.configure(results={"mem_save": text_result(saved)})

    result = result_of(provider.handle_tool_call("mem_save", {"title": "t", "content": "c"}))

    assert result == {**saved, "status": "🧠 saved #42"}
    state = read_state(str(hermes_home))
    assert (state["last_memory_id"], state["last_action"], state["project"]) == (
        42,
        "saved",
        PROJECT,
    )


@pytest.mark.parametrize(
    ("tool", "status", "action"),
    [
        ("mem_update", "🧠 updated #7", "updated"),
        ("mem_get_observation", "🧠 observation #7", "observation"),
        ("mem_pin", "🧠 pinned #7", "pinned"),
        ("mem_unpin", "🧠 unpinned #7", "unpinned"),
    ],
)
def test_id_from_the_arguments_when_the_result_has_none(
    make_provider: Any, fake_mcp: FakeMcp, hermes_home: Path, tool: str, status: str, action: str
) -> None:
    provider = make_provider()
    fake_mcp.configure(results={tool: text_result({"project": PROJECT, "result": "ok"})})

    result = result_of(provider.handle_tool_call(tool, {"id": 7, "expected_project": PROJECT}))

    assert result["status"] == status
    state = read_state(str(hermes_home))
    assert (state["last_memory_id"], state["last_action"]) == (7, action)


def test_refused_write_changes_no_state(
    make_provider: Any, fake_mcp: FakeMcp, hermes_home: Path
) -> None:
    provider = make_provider()
    fake_mcp.configure(
        results={"mem_update": text_result({"message": "owner mismatch"}, is_error=True)}
    )

    result = result_of(provider.handle_tool_call("mem_update", {"id": 7}))

    assert result == {"error": "owner mismatch"}
    assert read_state(str(hermes_home)) == {}


def test_confirmed_consolidation_keeps_upstream_status(
    make_provider: Any, fake_mcp: FakeMcp, hermes_home: Path
) -> None:
    provider = make_provider()
    body = {
        "status": "consolidated",
        "survivor": {"id": 9, "project": PROJECT},
        "soft_deleted_ids": [1, 2],
        "plan": {"fingerprint": "f"},
    }
    fake_mcp.configure(results={"mem_consolidate": text_result(body)})

    result = result_of(provider.handle_tool_call("mem_consolidate", {"ids_to_delete": [1, 2]}))

    assert result == {**body, "recall_status": "🧠 consolidated #9 (soft-deleted #1, #2)"}
    state = read_state(str(hermes_home))
    assert (state["last_memory_id"], state["last_action"]) == (9, "consolidated")


def test_finalize_suppresses_the_hook_end_and_the_next_write_resumes(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram, hermes_home: Path
) -> None:
    provider = make_provider()
    receipt = {"session_id": SESSION, "state": "finalized", "summary_id": 11}
    fake_mcp.configure(results={"mem_session_finalize": text_result(receipt)})

    result = result_of(provider.handle_tool_call("mem_session_finalize", {"summary": "s"}))
    provider.on_session_end([])
    assert provider._drain_for_test()
    fake_mcp.configure()
    provider.handle_tool_call("mem_save", {"title": "t", "content": "c"})

    assert args_of(fake_mcp, "mem_session_finalize") == [
        {"summary": "s", "session_id": SESSION, "project": PROJECT}
    ]
    assert result["status"] == "🧠 session finalized, summary #11"
    assert read_state(str(hermes_home))["last_action"] == "session_summary"
    assert end_posts(engram) == []
    assert args_of(fake_mcp, "mem_save")[0]["session_id"] == f"{SESSION}:resume:2"


@pytest.mark.parametrize(
    ("configure", "suppressed"),
    [
        ({"hang_on": ["mem_session_finalize"]}, True),
        ({"crash_on": ["mem_session_finalize"]}, True),
        (
            {
                "results": {
                    "mem_session_finalize": text_result(
                        {"error_code": "session_already_ended", "message": "ended"},
                        is_error=True,
                    )
                }
            },
            True,
        ),
        (
            {
                "results": {
                    "mem_session_finalize": text_result(
                        {"error_code": "unknown_project", "message": "no"}, is_error=True
                    )
                }
            },
            False,
        ),
    ],
)
def test_finalize_outcomes_decide_the_hook_end(
    make_provider: Any,
    fake_mcp: FakeMcp,
    engram: FakeEngram,
    monkeypatch: pytest.MonkeyPatch,
    configure: dict[str, Any],
    suppressed: bool,
) -> None:
    monkeypatch.setattr("hermes_recall.mcp_proxy.CALL_TIMEOUT_S", 0.5)
    provider = make_provider()
    fake_mcp.configure(**configure)

    provider.handle_tool_call("mem_session_finalize", {"summary": "s"})
    provider.on_session_end([])
    assert provider._drain_for_test()

    assert (end_posts(engram) == []) is suppressed


def test_finalize_for_another_project_leaves_recall_session_open(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    provider = make_provider()
    fake_mcp.configure(
        results={"mem_session_finalize": text_result({"state": "finalized", "summary_id": 3})}
    )

    provider.handle_tool_call("mem_session_finalize", {"summary": "s", "project": "other"})
    provider.on_session_end([])
    assert provider._drain_for_test()

    assert "session_id" not in args_of(fake_mcp, "mem_session_finalize")[0]
    assert len(end_posts(engram)) == 1


def test_private_tags_are_redacted_before_a_write_is_forwarded(
    make_provider: Any, fake_mcp: FakeMcp
) -> None:
    provider = make_provider()

    provider.handle_tool_call(
        "mem_save", {"title": "key <private>k1</private>", "content": "a <PRIVATE>s</PRIVATE> b"}
    )
    provider.handle_tool_call(
        "mem_judge_many",
        {"judgments": [{"judgment_id": "j", "reason": "<private>why</private>", "relation": "x"}]},
    )

    save = args_of(fake_mcp, "mem_save")[0]
    assert (save["title"], save["content"]) == ("key [REDACTED]", "a [REDACTED] b")
    judged = args_of(fake_mcp, "mem_judge_many")[0]
    assert judged["judgments"] == [{"judgment_id": "j", "reason": "[REDACTED]", "relation": "x"}]


def test_proxied_tool_results_are_never_passively_captured(
    make_provider: Any, engram: FakeEngram
) -> None:
    provider = make_provider()
    engram.route("POST", "/observations/passive", (200, {"extracted": 0}))
    long = "x" * 200
    messages = [
        {"role": "user", "content": "find the prompts"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "mem_search_prompts"}},
                {"id": "c2", "function": {"name": "terminal"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": long},
        {"role": "tool", "tool_call_id": "c2", "content": long + "y"},
    ]

    provider.sync_turn("find the prompts", "", messages=messages)
    assert provider._drain_for_test()

    sources = [
        r.json_body.get("source") for r in engram.requests if r.path == "/observations/passive"
    ]
    assert sources == ["terminal"]


def test_a_registered_session_costs_no_http_request(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram
) -> None:
    provider = make_provider()
    engram.reset()

    provider.handle_tool_call("mem_save", {"title": "t", "content": "c"})

    # The write goes to the MCP child; the session was registered at initialize.
    assert [r.path for r in engram.requests] == []
    assert len(args_of(fake_mcp, "mem_save")) == 1
