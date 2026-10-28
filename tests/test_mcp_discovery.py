"""Engram MCP tool discovery: the stdio handshake, the schema mapping and the schema cache.

The ``engram`` binary is :mod:`fake_mcp`. Never the real binary, never ``~/.engram``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from fake_mcp import AGENT_TOOL_NAMES, AGENT_TOOLS, FakeMcp, alive, wait_until

from hermes_recall import mcp_proxy


def test_discovery_handshakes_lists_and_stops_the_child(
    fake_mcp: FakeMcp, hermes_home: Path
) -> None:
    discovery = mcp_proxy.discover(str(fake_mcp.binary), str(hermes_home))

    assert discovery.error == ""
    assert discovery.version == "3.6.0"
    assert [tool["name"] for tool in discovery.tools] == AGENT_TOOL_NAMES
    starts = fake_mcp.starts()
    assert [start["argv"] for start in starts] == [["mcp", "--tools=agent"]]
    assert fake_mcp.methods() == ["initialize", "notifications/initialized", "tools/list"]
    init = fake_mcp.records()[1]["message"]
    assert init["params"]["clientInfo"]["name"] == "hermes-recall"
    # A discovery child never outlives the listing.
    assert wait_until(lambda: not alive(starts[0]["pid"]))


def test_schema_cache_file_skips_the_child_in_a_later_process(
    fake_mcp: FakeMcp, hermes_home: Path
) -> None:
    first = mcp_proxy.discover(str(fake_mcp.binary), str(hermes_home))
    cache = json.loads((hermes_home / "recall" / "mcp-tools.json").read_text("utf-8"))
    assert cache["version"] == "3.6.0"
    assert [tool["name"] for tool in cache["tools"]] == AGENT_TOOL_NAMES

    mcp_proxy.clear_discovery_cache()  # a new process
    again = mcp_proxy.discover(str(fake_mcp.binary), str(hermes_home))

    assert again.tools == first.tools
    assert again.version == "3.6.0"
    assert len(fake_mcp.starts()) == 1


def test_schema_cache_is_invalidated_when_the_binary_changes(
    fake_mcp: FakeMcp, hermes_home: Path
) -> None:
    mcp_proxy.discover(str(fake_mcp.binary), str(hermes_home))
    fake_mcp.configure(version="3.7.0", tools=AGENT_TOOLS[:2])
    # An upgrade replaces the binary: new bytes, new mtime.
    with open(fake_mcp.binary, "a", encoding="utf-8") as handle:
        handle.write("# upgraded\n")
    stamp = time.time() + 5
    os.utime(fake_mcp.binary, (stamp, stamp))
    mcp_proxy.clear_discovery_cache()

    again = mcp_proxy.discover(str(fake_mcp.binary), str(hermes_home))

    assert again.version == "3.7.0"
    assert [tool["name"] for tool in again.tools] == AGENT_TOOL_NAMES[:2]
    assert len(fake_mcp.starts()) == 2


@pytest.mark.parametrize(
    "behaviour",
    [{"crash_initialize": True}, {"fail_list": True}, {"hang_initialize": True}],
)
def test_discovery_failure_is_reported_and_never_cached_on_disk(
    fake_mcp: FakeMcp,
    hermes_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    behaviour: dict[str, Any],
) -> None:
    monkeypatch.setattr(mcp_proxy, "START_TIMEOUT_S", 1.0)
    fake_mcp.configure(**behaviour)

    started = time.monotonic()
    discovery = mcp_proxy.discover(str(fake_mcp.binary), str(hermes_home))

    assert time.monotonic() - started < 4.0
    assert discovery.tools == ()
    assert discovery.error
    assert not (hermes_home / "recall" / "mcp-tools.json").exists()
    assert all(wait_until(lambda s=s: not alive(s["pid"])) for s in fake_mcp.starts())


def test_to_hermes_schema_keeps_name_description_and_input_schema_only() -> None:
    upstream = next(tool for tool in AGENT_TOOLS if tool["name"] == "mem_save")

    schema = mcp_proxy.to_hermes_schema(upstream)

    assert schema == {
        "name": "mem_save",
        "description": upstream["description"],
        "parameters": upstream["inputSchema"],
    }
    assert schema is not None and schema["parameters"] is not upstream["inputSchema"]
    assert mcp_proxy.to_hermes_schema({"name": "mem_x"}) == {
        "name": "mem_x",
        "description": "",
        "parameters": {"type": "object", "properties": {}},
    }
    assert mcp_proxy.to_hermes_schema({"description": "nameless"}) is None


def test_a_missing_binary_is_reported_without_a_process(hermes_home: Path) -> None:
    discovery = mcp_proxy.discover(str(hermes_home / "no-engram"), str(hermes_home))

    assert discovery.tools == ()
    assert "not found" in discovery.error
