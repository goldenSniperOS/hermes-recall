"""A fake ``engram`` binary that speaks Engram's stdio MCP, for the proxy tests.

Run as a script (through the wrapper :class:`FakeMcp` writes) it answers
``instance-id`` like the real binary and serves ``mcp --tools=agent``: newline-
delimited JSON-RPC with ``initialize``, ``notifications/initialized``,
``tools/list`` and ``tools/call``. Every message it reads is appended to a JSON-lines
log, so a test can assert exactly what the provider sent.

Behaviour comes from a JSON config file that is re-read on every message, so a test
can change it between two calls of the same child:

* ``tools``: the ``tools/list`` answer (default :data:`AGENT_TOOLS`).
* ``version``: ``serverInfo.version`` (default ``3.6.0``).
* ``results``: tool name -> MCP result object returned by ``tools/call``. Without
  one the call echoes ``{"tool", "arguments", "result": "ok"}`` as JSON text.
* ``hang_on`` / ``crash_on`` / ``rpc_error_on``: tool names that never answer, make
  the process exit, or get a JSON-RPC error.
* ``hang_initialize``, ``crash_initialize``, ``fail_list``: break the handshake.

Imported as a module it provides :class:`FakeMcp` and :data:`AGENT_TOOLS`. Never
runs the real ``engram`` and never opens ``~/.engram``.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

FAKE_INSTANCE_ID = "0123456789abcdef0123456789abcdef"

# Engram 3.6.0 ``engram mcp --tools=agent``: names, parameters and required lists as
# the real binary lists them (descriptions shortened). Two tools keep the extra
# ``annotations`` / ``defer_loading`` keys the real list carries.
_AGENT_SPEC: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("mem_capture_passive", ("content", "session_id", "source"), ("content",)),
    (
        "mem_compare",
        ("confidence", "memory_id_a", "memory_id_b", "model", "reasoning", "relation"),
        ("memory_id_a", "memory_id_b", "relation", "confidence", "reasoning"),
    ),
    (
        "mem_consolidate",
        (
            "confirm",
            "expected_project",
            "fingerprint",
            "ids_to_delete",
            "new_content",
            "new_title",
            "topic_key",
            "type",
        ),
        ("ids_to_delete", "new_title", "new_content", "expected_project"),
    ),
    ("mem_context", ("compact", "max_bytes", "older_history", "project", "scope"), ()),
    ("mem_current_project", ("cwd", "directory"), ()),
    ("mem_doctor", ("check", "project"), ()),
    ("mem_get_observation", ("history_cursor", "id", "include_history", "project"), ("id",)),
    (
        "mem_judge",
        ("confidence", "evidence", "judgment_id", "reason", "relation", "session_id"),
        ("judgment_id", "relation"),
    ),
    ("mem_judge_many", ("judgments", "session_id"), ("judgments",)),
    ("mem_list_projects", (), ()),
    ("mem_pin", ("id",), ("id",)),
    ("mem_review", ("action", "id", "limit", "observation_id", "project"), ("action",)),
    (
        "mem_save",
        (
            "capture_prompt",
            "content",
            "observation",
            "project",
            "project_choice_reason",
            "recovery_token",
            "scope",
            "session_id",
            "title",
            "topic_key",
            "type",
        ),
        ("title",),
    ),
    (
        "mem_save_prompt",
        ("content", "project", "project_choice_reason", "recovery_token", "session_id"),
        ("content",),
    ),
    (
        "mem_search",
        (
            "all_projects",
            "limit",
            "match_mode",
            "project",
            "query",
            "response_format",
            "scope",
            "type",
        ),
        ("query",),
    ),
    ("mem_search_prompts", ("all_projects", "limit", "project", "query"), ("query",)),
    ("mem_session_end", ("id", "summary"), ("id",)),
    (
        "mem_session_finalize",
        ("project", "project_choice_reason", "recovery_token", "session_id", "summary"),
        ("session_id", "summary"),
    ),
    ("mem_session_start", ("directory", "id"), ("id",)),
    (
        "mem_session_summary",
        ("content", "project", "project_choice_reason", "recovery_token", "session_id"),
        ("content",),
    ),
    ("mem_suggest_topic_key", ("content", "title", "type"), ()),
    ("mem_unpin", ("id",), ("id",)),
    (
        "mem_update",
        (
            "content",
            "expected_project",
            "find",
            "id",
            "replace",
            "scope",
            "title",
            "topic_key",
            "type",
        ),
        ("id", "expected_project"),
    ),
)


def _tool(name: str, props: tuple[str, ...], required: tuple[str, ...]) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "name": name,
        "description": f"Engram {name} (fake).",
        "inputSchema": {
            "type": "object",
            "properties": {prop: {"type": "string"} for prop in props},
            "required": list(required),
        },
    }
    if name in ("mem_save", "mem_session_end"):
        tool["annotations"] = {"readOnlyHint": False}
        tool["defer_loading"] = True
    return tool


AGENT_TOOLS: list[dict[str, Any]] = [_tool(*spec) for spec in _AGENT_SPEC]
AGENT_TOOL_NAMES: list[str] = [tool["name"] for tool in AGENT_TOOLS]


# -- the child process -------------------------------------------------------------


def _config() -> dict[str, Any]:
    path = os.environ.get("FAKE_MCP_CONFIG", "")
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _record(entry: dict[str, Any]) -> None:
    path = os.environ.get("FAKE_MCP_LOG", "")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _hang() -> None:
    while True:
        time.sleep(60)


def _call_result(config: dict[str, Any], name: str, arguments: Any) -> dict[str, Any]:
    results = config.get("results") or {}
    if name in results:
        return results[name]
    text = json.dumps({"tool": name, "arguments": arguments, "result": "ok"})
    return {"content": [{"type": "text", "text": text}]}


def _serve() -> None:
    _record({"event": "start", "argv": sys.argv[1:], "cwd": os.getcwd(), "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        _record({"event": "message", "message": message})
        config = _config()
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            if config.get("hang_initialize"):
                _hang()
            if config.get("crash_initialize"):
                os._exit(3)
            version = config.get("version", "3.6.0")
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": True}},
                        "serverInfo": {"name": "engram", "version": version},
                    },
                }
            )
        elif method == "tools/list":
            if config.get("fail_list"):
                _send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -1, "message": "no"}})
                continue
            tools = config.get("tools", AGENT_TOOLS)
            _send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            if name in (config.get("hang_on") or []):
                _hang()
            if name in (config.get("crash_on") or []):
                os._exit(4)
            if name in (config.get("rpc_error_on") or []):
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32602, "message": f"tool '{name}' not found"},
                    }
                )
                continue
            # A notification first, as a real server may interleave one.
            _send({"jsonrpc": "2.0", "method": "notifications/message", "params": {}})
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": _call_result(config, name, params.get("arguments")),
                }
            )
        elif request_id is not None:
            _send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    _record({"event": "eof"})


def main() -> None:
    args = sys.argv[1:]
    if args == ["instance-id"]:
        print(os.environ.get("FAKE_ENGRAM_ID", FAKE_INSTANCE_ID))
        return
    if args[:1] == ["mcp"]:
        _serve()
        return
    _record({"event": "other", "argv": args})
    sys.exit(1)


# -- the test-side helper ----------------------------------------------------------


def alive(pid: int) -> bool:
    """Whether process ``pid`` still runs (a reaped or zombie child does not)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return waited == 0


def wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` passes; its last answer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


_WRAPPER = """#!{python}
import os, runpy, sys
os.environ.setdefault("FAKE_MCP_CONFIG", {config!r})
os.environ.setdefault("FAKE_MCP_LOG", {log!r})
runpy.run_path({script!r}, run_name="__main__")
"""


class FakeMcp:
    """An executable fake ``engram`` in ``directory`` plus its config and request log."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.binary = directory / "engram"
        self.config_path = directory / "fake-mcp.json"
        self.log_path = directory / "fake-mcp.log"
        self.binary.write_text(
            _WRAPPER.format(
                python=sys.executable,
                config=str(self.config_path),
                log=str(self.log_path),
                script=str(Path(__file__).resolve()),
            ),
            encoding="utf-8",
        )
        mode = self.binary.stat().st_mode
        self.binary.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.configure()

    def configure(self, **config: Any) -> None:
        """Replace the child's behaviour (read by the child on its next message)."""
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def records(self) -> list[dict[str, Any]]:
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return [json.loads(line) for line in lines if line.strip()]

    def starts(self) -> list[dict[str, Any]]:
        """One entry per ``engram mcp`` process launched (argv, cwd, pid)."""
        return [r for r in self.records() if r["event"] == "start"]

    def methods(self) -> list[str]:
        return [
            r["message"].get("method", "")
            for r in self.records()
            if r["event"] == "message" and "method" in r["message"]
        ]

    def calls(self) -> list[tuple[str, Any]]:
        """``(tool name, arguments)`` of every ``tools/call`` the child received."""
        return [
            (r["message"]["params"]["name"], r["message"]["params"].get("arguments"))
            for r in self.records()
            if r["event"] == "message" and r["message"].get("method") == "tools/call"
        ]


if __name__ == "__main__":
    main()
