"""S14: when the proxy cannot serve, the v0.4 HTTP tools are served unchanged; status says which.

The ``engram`` binary is :mod:`fake_mcp`; the HTTP server a
:class:`~fake_engram.FakeEngram`. Never the real binary, never 127.0.0.1:7437, never
``~/.engram``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_engram import FakeEngram, RecordedRequest
from fake_mcp import AGENT_TOOL_NAMES, FakeMcp
from test_cli import load_root_cli

from hermes_recall import mcp_proxy
from hermes_recall import tools as tools_module
from hermes_recall.config import RecallConfig
from hermes_recall.format import render_status_line
from hermes_recall.provider import RecallProvider
from hermes_recall.server import FOREIGN, ServerStatus

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
        "autostart": False,
    }
    values.update(over)
    return RecallConfig(**values)


def result_of(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def echo_session(request: RecordedRequest) -> tuple[int, Any]:
    return (201, {"id": request.json_body["id"], "status": "created"})


def posts(fake: FakeEngram, path: str) -> list[RecordedRequest]:
    return [r for r in fake.requests if r.method == "POST" and r.path == path]


@pytest.fixture
def engram(fake_engram: FakeEngram) -> FakeEngram:
    fake_engram.route("GET", "/health", (200, {"status": "ok"}))
    fake_engram.route("GET", "/project/current", (200, {"project": PROJECT}))
    fake_engram.route("POST", "/sessions", echo_session)
    fake_engram.route("POST", "/observations", (201, {"id": 900, "status": "saved"}))
    fake_engram.route("POST", "/prompts", (201, {"id": 1}))
    fake_engram.route("GET", "/search", (200, []))
    fake_engram.route("GET", "/context", (200, {"context": ""}))
    fake_engram.route("POST", f"/sessions/{SESSION}/end", (200, {"status": "ended"}))
    return fake_engram


@pytest.fixture
def local_default(engram: FakeEngram, monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the FakeEngram's URL the local default address (what the proxy requires)."""
    monkeypatch.setenv("ENGRAM_PORT", str(engram.port))
    return engram.base_url


@pytest.fixture
def make_provider(hermes_home: Path) -> Iterator[Any]:
    built: list[RecallProvider] = []

    def factory(url: str, *, agent_context: str = "primary", **over: Any) -> RecallProvider:
        provider = RecallProvider(config=make_config(url, **over))
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


def assert_v04_tools(provider: RecallProvider, fake: FakeMcp | None = None) -> None:
    assert provider.get_tool_schemas() == tools_module.get_tool_schemas()
    if fake is not None:
        assert fake.calls() == []


# -- S14 the conditions ---------------------------------------------------------------


def test_engram_url_set_serves_the_http_tools_without_a_child(
    make_provider: Any,
    fake_mcp: FakeMcp,
    local_default: str,
    hermes_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENGRAM_URL", local_default)
    provider = make_provider(local_default)

    assert_v04_tools(provider, fake_mcp)
    assert fake_mcp.starts() == []
    recorded = mcp_proxy.read_tool_source(str(hermes_home))
    assert recorded is not None and recorded["source"] == "http"
    assert "ENGRAM_URL" in recorded["reason"]


def test_a_non_default_url_serves_the_http_tools_without_a_child(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram, hermes_home: Path
) -> None:
    # ENGRAM_PORT stays at the conftest's 1, so the fake's port is not the default.
    provider = make_provider(engram.base_url)

    assert_v04_tools(provider, fake_mcp)
    assert fake_mcp.starts() == []
    reason = mcp_proxy.read_tool_source(str(hermes_home))["reason"]
    assert engram.base_url in reason
    assert "local" in reason


def test_a_missing_binary_serves_the_http_tools(
    make_provider: Any, local_default: str, hermes_home: Path
) -> None:
    provider = make_provider(local_default)  # conftest's ENGRAM_BIN does not exist

    assert_v04_tools(provider)
    assert "not found" in mcp_proxy.read_tool_source(str(hermes_home))["reason"]


@pytest.mark.parametrize("behaviour", [{"crash_initialize": True}, {"fail_list": True}])
def test_a_child_that_cannot_list_serves_the_http_tools(
    make_provider: Any,
    fake_mcp: FakeMcp,
    local_default: str,
    hermes_home: Path,
    behaviour: dict[str, Any],
) -> None:
    fake_mcp.configure(**behaviour)
    provider = make_provider(local_default)

    assert_v04_tools(provider, fake_mcp)
    assert len(fake_mcp.starts()) == 1  # the failed discovery, nothing after it
    assert mcp_proxy.read_tool_source(str(hermes_home))["source"] == "http"


def test_the_proxy_is_used_only_when_every_condition_holds(
    make_provider: Any, fake_mcp: FakeMcp, local_default: str, hermes_home: Path
) -> None:
    provider = make_provider(local_default)

    assert [s["name"] for s in provider.get_tool_schemas()] == AGENT_TOOL_NAMES
    recorded = mcp_proxy.read_tool_source(str(hermes_home))
    assert recorded is not None
    assert (recorded["source"], recorded["count"], recorded["version"]) == ("mcp", 23, "3.6.0")


# -- the fallback is the v0.4 tool set, behaviour and all ------------------------------------


def test_fallback_tool_calls_go_over_http_exactly_as_in_v04(
    make_provider: Any, engram: FakeEngram, hermes_home: Path
) -> None:
    provider = make_provider(engram.base_url)

    result = result_of(provider.handle_tool_call("mem_save", {"title": "t", "content": "c"}))

    assert result["id"] == 900
    assert result["status"] == "🧠 saved #900"
    body = posts(engram, "/observations")[0].json_body
    assert (body["session_id"], body["project"], body["title"]) == (SESSION, PROJECT, "t")


def test_fallback_still_refuses_non_primary_writes(make_provider: Any, engram: FakeEngram) -> None:
    provider = make_provider(engram.base_url, agent_context="subagent")
    engram.reset()

    result = result_of(provider.handle_tool_call("mem_save", {"title": "t", "content": "c"}))

    assert "writes are disabled" in result["error"]
    assert engram.requests == []


def test_fallback_finalize_still_suppresses_the_hook_end(
    make_provider: Any, engram: FakeEngram
) -> None:
    engram.route(
        "POST",
        f"/sessions/{SESSION}/finalize",
        (200, {"session_id": SESSION, "state": "finalized", "summary_id": 5}),
    )
    provider = make_provider(engram.base_url)

    provider.handle_tool_call("mem_session_finalize", {"summary": "done"})
    provider.on_session_end([])
    assert provider._drain_for_test()

    assert posts(engram, f"/sessions/{SESSION}/end") == []


def test_fallback_allowlist_filters_the_http_tools(make_provider: Any, engram: FakeEngram) -> None:
    provider = make_provider(engram.base_url, tools=("mem_search", "mem_timeline"))

    assert [s["name"] for s in provider.get_tool_schemas()] == ["mem_search", "mem_timeline"]
    assert "unknown tool" in result_of(provider.handle_tool_call("mem_save", {}))["error"]


def test_lifecycle_hooks_stay_on_http_when_the_proxy_serves_tools(
    make_provider: Any, fake_mcp: FakeMcp, engram: FakeEngram, local_default: str
) -> None:
    provider = make_provider(local_default)
    assert provider._tool_catalog().source == "mcp"

    provider.on_turn_start(1, "a real question about the loader")
    provider.prefetch("a real question about the loader")
    provider.on_session_end([])
    assert provider._drain_for_test()

    assert [r.json_body["id"] for r in posts(engram, "/sessions")] == [SESSION]
    assert len(posts(engram, "/prompts")) == 1
    assert any(r.path == "/search" for r in engram.requests)
    assert len(posts(engram, f"/sessions/{SESSION}/end")) == 1
    # Hooks never start the MCP child: only the discovery child ever ran.
    assert len(fake_mcp.starts()) == 1


def test_a_foreign_server_refuses_proxied_calls_without_the_child(
    make_provider: Any, fake_mcp: FakeMcp, local_default: str
) -> None:
    provider = make_provider(local_default)
    provider._server_status = ServerStatus(FOREIGN, "different instance", local_default)

    result = result_of(provider.handle_tool_call("mem_search", {"query": "x"}))

    assert "foreign instance" in result["error"]
    assert fake_mcp.calls() == []


# -- status ------------------------------------------------------------------------------


def test_render_status_line_shows_the_tool_source_only_when_given() -> None:
    assert "tools:" not in render_status_line("p", server_ok=True)
    text = render_status_line("p", server_ok=True, tools="Engram MCP, 23 tools (Engram 3.6.0)")
    assert "  tools:        Engram MCP, 23 tools (Engram 3.6.0)" in text.splitlines()


def test_recall_command_names_the_proxy(
    make_provider: Any, fake_mcp: FakeMcp, local_default: str
) -> None:
    provider = make_provider(local_default)

    text = provider.recall_command("")

    assert "  tools:        Engram MCP, 23 tools (Engram 3.6.0)" in text.splitlines()


def test_recall_command_names_the_fallback_and_why(make_provider: Any, engram: FakeEngram) -> None:
    provider = make_provider(engram.base_url)

    line = next(x for x in provider.recall_command("").splitlines() if "tools:" in x)

    assert line.startswith("  tools:        HTTP fallback, 13 tools: ")
    assert engram.base_url in line


def test_cli_status_reads_the_recorded_tool_source(
    make_provider: Any,
    fake_mcp: FakeMcp,
    local_default: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_provider(local_default)  # the agent process records its decision
    cli = load_root_cli()
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)

    assert cli.recall_command(parser.parse_args(["status"])) == 0

    out = capsys.readouterr().out
    assert "  tools:        Engram MCP, 23 tools (Engram 3.6.0)" in out.splitlines()


def test_cli_status_without_a_record_says_so(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ENGRAM_URL", "http://127.0.0.1:1")
    cli = load_root_cli()
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)

    cli.recall_command(parser.parse_args(["status"]))

    assert "  tools:        — (not recorded yet)" in capsys.readouterr().out.splitlines()
