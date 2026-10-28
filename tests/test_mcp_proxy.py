"""The Engram MCP proxy: tool discovery, schema cache, ``tools/call`` dispatch, child lifecycle.

The ``engram`` binary is :mod:`fake_mcp` (``ENGRAM_BIN``), the HTTP server a
:class:`~fake_engram.FakeEngram` on an ephemeral port that ``ENGRAM_PORT`` names, so the
configured URL is "the local default" the proxy requires. Never the real binary, never
127.0.0.1:7437, never ``~/.engram``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_engram import FakeEngram
from fake_mcp import AGENT_TOOL_NAMES, AGENT_TOOLS, FakeMcp, alive, wait_until

from hermes_recall import mcp_proxy
from hermes_recall.config import RecallConfig
from hermes_recall.provider import RecallProvider

PROJECT = "hermes-recall"
SESSION = "sess-1"


def make_config(url: str, **over: Any) -> RecallConfig:
    values: dict[str, Any] = {
        "url": url,
        "project_override": "",
        "recall_limit": 5,
        "recall_max_chars": 2500,
        "enabled": True,
        "auth_token": "",
        # The fake binary cannot serve HTTP; the FakeEngram is the server.
        "autostart": False,
    }
    values.update(over)
    return RecallConfig(**values)


def echo_session(request: Any) -> tuple[int, Any]:
    return (201, {"id": request.json_body["id"], "status": "created"})


def wire(fake: FakeEngram) -> None:
    fake.route("GET", "/project/current", (200, {"project": PROJECT}))
    fake.route("POST", "/sessions", echo_session)


def result_of(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


@pytest.fixture
def engram(fake_engram: FakeEngram, monkeypatch: pytest.MonkeyPatch) -> FakeEngram:
    """The HTTP side, on the port ``ENGRAM_PORT`` names (the proxy's local default)."""
    monkeypatch.setenv("ENGRAM_PORT", str(fake_engram.port))
    wire(fake_engram)
    return fake_engram


@pytest.fixture
def make_provider(hermes_home: Path, engram: FakeEngram) -> Iterator[Any]:
    built: list[RecallProvider] = []

    def factory(*, agent_context: str = "primary", initialize: bool = True, **over: Any):
        provider = RecallProvider(config=make_config(engram.base_url, **over))
        # The host asks for schemas in add_provider, BEFORE initialize.
        provider.get_tool_schemas()
        if initialize:
            provider.initialize(
                SESSION,
                hermes_home=str(hermes_home),
                platform="cli",
                agent_context=agent_context,
            )
            assert provider._drain_for_test()
        built.append(provider)
        return provider

    yield factory
    for provider in built:
        provider.shutdown()


# -- S9 schemas -----------------------------------------------------------------


def test_upstream_tools_map_to_hermes_schemas_unchanged(
    fake_mcp: FakeMcp, make_provider: Any
) -> None:
    provider = make_provider()

    schemas = provider.get_tool_schemas()

    assert [schema["name"] for schema in schemas] == AGENT_TOOL_NAMES
    save = next(schema for schema in schemas if schema["name"] == "mem_save")
    upstream = next(tool for tool in AGENT_TOOLS if tool["name"] == "mem_save")
    # Exactly the Hermes shape: annotations / defer_loading are not forwarded.
    assert save == {
        "name": "mem_save",
        "description": upstream["description"],
        "parameters": upstream["inputSchema"],
    }


def test_get_tool_schemas_is_cached_and_side_effect_free(
    fake_mcp: FakeMcp, make_provider: Any, engram: FakeEngram
) -> None:
    provider = make_provider(initialize=False)
    engram.reset()

    first = provider.get_tool_schemas()
    first[0]["name"] = "mutated"
    for _ in range(3):
        assert provider.get_tool_schemas()[0]["name"] == AGENT_TOOL_NAMES[0]

    # One discovery child for the whole provider, no HTTP traffic, no call child.
    assert len(fake_mcp.starts()) == 1
    assert engram.requests == []


def test_tools_allowlist_keeps_only_the_named_tools_in_upstream_order(
    fake_mcp: FakeMcp, make_provider: Any
) -> None:
    provider = make_provider(tools=("mem_save", "mem_search", "not_a_tool"))

    names = [schema["name"] for schema in provider.get_tool_schemas()]

    assert names == ["mem_save", "mem_search"]
    refused = result_of(provider.handle_tool_call("mem_context", {}))
    assert "unknown tool" in refused["error"]
    assert fake_mcp.calls() == []


# -- S10 dispatch ----------------------------------------------------------------


def test_tool_call_is_forwarded_and_the_upstream_json_returned(
    fake_mcp: FakeMcp, make_provider: Any, engram: FakeEngram
) -> None:
    provider = make_provider()
    fake_mcp.configure(
        results={
            "mem_pin": {
                "content": [
                    {"type": "text", "text": json.dumps({"id": 5, "pinned": True, "result": "ok"})}
                ]
            }
        }
    )

    result = result_of(provider.handle_tool_call("mem_pin", {"id": 5}))

    assert result["id"] == 5
    assert result["pinned"] is True
    assert fake_mcp.calls() == [("mem_pin", {"id": 5})]
    calls = fake_mcp.starts()
    # Discovery child, then one long-lived call child.
    assert len(calls) == 2


def test_one_child_serves_every_call(fake_mcp: FakeMcp, make_provider: Any) -> None:
    provider = make_provider()

    for observation_id in (1, 2, 3):
        provider.handle_tool_call("mem_pin", {"id": observation_id})

    assert [args["id"] for _, args in fake_mcp.calls()] == [1, 2, 3]
    assert len(fake_mcp.starts()) == 2
    call_child = fake_mcp.starts()[1]
    assert call_child["argv"] == ["mcp", "--tools=agent"]


def test_structured_content_wins_over_text(fake_mcp: FakeMcp, make_provider: Any) -> None:
    provider = make_provider()
    fake_mcp.configure(
        results={
            "mem_list_projects": {
                "content": [{"type": "text", "text": "projects: a, b"}],
                "structuredContent": {"projects": ["a", "b"]},
            }
        }
    )

    assert result_of(provider.handle_tool_call("mem_list_projects", {})) == {"projects": ["a", "b"]}


def test_plain_text_result_is_wrapped(fake_mcp: FakeMcp, make_provider: Any) -> None:
    provider = make_provider()
    fake_mcp.configure(
        results={
            "mem_suggest_topic_key": {
                "content": [
                    {"type": "text", "text": "Suggested topic_key:"},
                    {"type": "text", "text": "architecture/auth"},
                ]
            }
        }
    )

    result = result_of(provider.handle_tool_call("mem_suggest_topic_key", {"title": "x"}))

    assert result == {"result": "Suggested topic_key:\narchitecture/auth"}


def test_upstream_is_error_becomes_an_error_with_its_fields(
    fake_mcp: FakeMcp, make_provider: Any
) -> None:
    provider = make_provider()
    refusal = {
        "available_projects": ["proj"],
        "error_code": "unknown_project",
        "hint": "Use one of the available_projects values.",
        "message": 'Project "other" is not backed by known context.',
    }
    fake_mcp.configure(
        results={
            "mem_list_projects": {
                "content": [{"type": "text", "text": json.dumps(refusal)}],
                "isError": True,
            },
            "mem_pin": {"content": [{"type": "text", "text": "id is required"}], "isError": True},
        }
    )

    listed = result_of(provider.handle_tool_call("mem_list_projects", {}))
    pinned = result_of(provider.handle_tool_call("mem_pin", {}))

    assert listed == {
        "error": refusal["message"],
        "available_projects": ["proj"],
        "error_code": "unknown_project",
        "hint": refusal["hint"],
    }
    assert pinned == {"error": "id is required"}


def test_json_rpc_error_becomes_an_error(fake_mcp: FakeMcp, make_provider: Any) -> None:
    provider = make_provider()
    fake_mcp.configure(rpc_error_on=["mem_pin"])

    result = result_of(provider.handle_tool_call("mem_pin", {"id": 1}))

    assert result == {"error": "Engram MCP refused mem_pin: tool 'mem_pin' not found"}
    # A protocol refusal is not a dead child: the next call reuses it.
    fake_mcp.configure()
    provider.handle_tool_call("mem_pin", {"id": 2})
    assert len(fake_mcp.starts()) == 2


@pytest.mark.parametrize("args", ["{not json", "[1, 2]", 7])
def test_bad_arguments_never_reach_the_child(
    fake_mcp: FakeMcp, make_provider: Any, args: Any
) -> None:
    provider = make_provider()

    result = result_of(provider.handle_tool_call("mem_pin", args))

    assert result == {"error": "arguments must be a JSON object"}
    assert fake_mcp.calls() == []


def test_json_string_arguments_are_accepted(fake_mcp: FakeMcp, make_provider: Any) -> None:
    provider = make_provider()

    provider.handle_tool_call("mem_pin", '{"id": 9}')
    provider.handle_tool_call("mem_list_projects", None)

    assert fake_mcp.calls() == [("mem_pin", {"id": 9}), ("mem_list_projects", {})]


def test_call_child_runs_in_the_working_directory_with_a_stderr_log(
    fake_mcp: FakeMcp,
    make_provider: Any,
    hermes_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workdir = tmp_path / "repo"
    workdir.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(workdir))
    provider = make_provider()

    provider.handle_tool_call("mem_list_projects", {})

    call_child = fake_mcp.starts()[-1]
    assert Path(call_child["cwd"]).resolve() == workdir.resolve()
    assert (hermes_home / "recall" / "engram-mcp.log").exists()


# -- S10 timeout, crash, restart --------------------------------------------------


def test_a_hung_call_times_out_kills_the_child_and_restarts_once(
    fake_mcp: FakeMcp, make_provider: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_proxy, "CALL_TIMEOUT_S", 0.5)
    provider = make_provider()
    fake_mcp.configure(hang_on=["mem_pin"])

    started = time.monotonic()
    hung = result_of(provider.handle_tool_call("mem_pin", {"id": 1}))

    assert time.monotonic() - started < 3.0
    assert "did not answer mem_pin" in hung["error"]
    first_child = fake_mcp.starts()[1]
    assert wait_until(lambda: not alive(first_child["pid"]))

    # The single restart: a fresh child answers the next call.
    ok = result_of(provider.handle_tool_call("mem_list_projects", {}))
    assert ok["tool"] == "mem_list_projects"
    assert len(fake_mcp.starts()) == 3

    # A second death spends nothing more: no third call child is ever started.
    provider.handle_tool_call("mem_pin", {"id": 2})
    refused = result_of(provider.handle_tool_call("mem_list_projects", {}))
    assert "restart" in refused["error"]
    assert len(fake_mcp.starts()) == 3


def test_a_crash_mid_call_is_an_error_and_the_next_call_restarts(
    fake_mcp: FakeMcp, make_provider: Any
) -> None:
    provider = make_provider()
    fake_mcp.configure(crash_on=["mem_pin"])

    crashed = result_of(provider.handle_tool_call("mem_pin", {"id": 1}))

    assert "exited during mem_pin" in crashed["error"]
    fake_mcp.configure()
    assert result_of(provider.handle_tool_call("mem_pin", {"id": 2}))["tool"] == "mem_pin"
    assert len(fake_mcp.starts()) == 3


def test_a_child_that_died_between_calls_is_restarted_before_sending(
    fake_mcp: FakeMcp, make_provider: Any
) -> None:
    provider = make_provider()
    provider.handle_tool_call("mem_list_projects", {})
    os.kill(fake_mcp.starts()[1]["pid"], 9)
    assert wait_until(lambda: provider._mcp is not None and not provider._mcp.child_alive())

    result = result_of(provider.handle_tool_call("mem_pin", {"id": 3}))

    assert result["tool"] == "mem_pin"
    assert len(fake_mcp.starts()) == 3


def test_a_hung_call_never_wedges_the_lock_for_another_thread(
    fake_mcp: FakeMcp, make_provider: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_proxy, "CALL_TIMEOUT_S", 0.8)
    provider = make_provider()
    provider.handle_tool_call("mem_list_projects", {})
    fake_mcp.configure(hang_on=["mem_pin"])
    results: dict[str, Any] = {}

    def hung() -> None:
        results["hung"] = result_of(provider.handle_tool_call("mem_pin", {"id": 1}))

    thread = threading.Thread(target=hung)
    thread.start()
    assert wait_until(lambda: ("mem_pin", {"id": 1}) in fake_mcp.calls())
    # Queued behind the hung call, then served by the restarted child.
    results["second"] = result_of(provider.handle_tool_call("mem_list_projects", {}))
    thread.join(5)

    assert "did not answer" in results["hung"]["error"]
    assert results["second"]["tool"] == "mem_list_projects"


def test_concurrent_calls_are_serialized_on_one_child(
    fake_mcp: FakeMcp, make_provider: Any
) -> None:
    provider = make_provider()
    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    def call(index: int) -> None:
        payload = result_of(provider.handle_tool_call("mem_pin", {"id": index}))
        with lock:
            results.append(payload)

    threads = [threading.Thread(target=call, args=(index,)) for index in range(1, 7)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert sorted(r["arguments"]["id"] for r in results) == [1, 2, 3, 4, 5, 6]
    assert len(fake_mcp.starts()) == 2


# -- S10 shutdown ------------------------------------------------------------------


def test_shutdown_stops_the_child_and_never_the_server(
    fake_mcp: FakeMcp, make_provider: Any, engram: FakeEngram
) -> None:
    provider = make_provider()
    provider.handle_tool_call("mem_list_projects", {})
    call_child = fake_mcp.starts()[1]
    engram.reset()

    provider.shutdown()

    assert wait_until(lambda: not alive(call_child["pid"]))
    after = result_of(provider.handle_tool_call("mem_list_projects", {}))
    assert "error" in after
    assert len(fake_mcp.starts()) == 2
    # Nothing asked the HTTP server to stop (there is no such route to ask).
    assert [r.path for r in engram.requests if r.method == "POST"] == []


def test_shutdown_during_a_hung_call_unblocks_it(
    fake_mcp: FakeMcp, make_provider: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_proxy, "CALL_TIMEOUT_S", 30.0)
    provider = make_provider()
    fake_mcp.configure(hang_on=["mem_pin"])
    results: dict[str, Any] = {}
    thread = threading.Thread(
        target=lambda: results.update(r=provider.handle_tool_call("mem_pin", {"id": 1}))
    )
    thread.start()
    assert wait_until(lambda: ("mem_pin", {"id": 1}) in fake_mcp.calls())

    started = time.monotonic()
    provider.shutdown()
    thread.join(5)

    assert time.monotonic() - started < 5.0
    assert "error" in result_of(results["r"])


def test_shutdown_without_a_child_is_a_no_op(fake_mcp: FakeMcp, make_provider: Any) -> None:
    provider = make_provider()

    provider.shutdown()
    provider.shutdown()

    assert len(fake_mcp.starts()) == 1
