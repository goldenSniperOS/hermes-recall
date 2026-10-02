"""Engram server auto-start (``hermes_recall.server``) and its provider wiring.

Nothing here touches the user's Engram on 127.0.0.1:7437 or runs the real
``engram`` binary. Auto-start only acts when the configured URL is the address a
plain ``engram serve`` would bind, which follows ``ENGRAM_PORT``; every test that
needs a match sets ``ENGRAM_PORT`` to a free port of its own and points
``ENGRAM_BIN`` at a fake executable written into ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_engram import FakeEngram

from hermes_recall import provider as provider_module
from hermes_recall import server as server_module
from hermes_recall.client import EngramClient
from hermes_recall.config import RecallConfig
from hermes_recall.format import render_status_line
from hermes_recall.provider import RecallProvider
from hermes_recall.server import (
    ADOPTED,
    DISABLED,
    EXTERNAL,
    FOREIGN,
    STARTED,
    UNAVAILABLE,
    ServerStatus,
    ensure_server,
    expected_serve_port,
    is_spawnable_url,
    read_server_state,
    write_server_state,
)

INSTANCE_ID = "0123456789abcdef0123456789abcdef"
OTHER_ID = "fedcba9876543210fedcba9876543210"

# The fake binary: ``instance-id`` prints FAKE_ENGRAM_ID (an update notice on stderr,
# like the real 2.0.0 binary); ``serve`` records how it was launched and answers
# /health on ENGRAM_PORT until FAKE_ENGRAM_LIFETIME runs out.
FAKE_BINARY = """#!{python}
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
if args == ["instance-id"]:
    sys.stderr.write("Update available: 9.9.9\\n")
    sys.stdout.write(os.environ.get("FAKE_ENGRAM_ID", "") + "\\n")
    sys.exit(int(os.environ.get("FAKE_ENGRAM_ID_EXIT", "0")))
if args != ["serve"]:
    sys.stderr.write("unknown command\\n")
    sys.exit(2)

marker = os.environ.get("FAKE_ENGRAM_MARKER", "")
if marker:
    with open(marker, "w") as handle:
        json.dump({{"argv": sys.argv, "pid": os.getpid(), "sid": os.getsid(0),
                    "cwd": os.getcwd()}}, handle)

serve_id = os.environ.get("FAKE_ENGRAM_SERVE_ID", os.environ.get("FAKE_ENGRAM_ID", ""))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        body = json.dumps({{"status": "ok", "service": "engram", "version": "2.0.0",
                            "instance_id": serve_id}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = HTTPServer(("127.0.0.1", int(os.environ["ENGRAM_PORT"])), Handler)
server.timeout = 0.1
stop_at = time.monotonic() + float(os.environ.get("FAKE_ENGRAM_LIFETIME", "20"))
while time.monotonic() < stop_at:
    server.handle_request()
"""


# -- helpers ------------------------------------------------------------------


def free_port() -> int:
    """A port nothing listens on (bound and released by the OS)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_config(url: str, **over: Any) -> RecallConfig:
    values: dict[str, Any] = {
        "url": url,
        "project_override": "",
        "recall_limit": 5,
        "recall_max_chars": 2500,
        "enabled": True,
        "auth_token": "",
        "autostart": True,
    }
    values.update(over)
    return RecallConfig(**values)


def write_fake_binary(directory: Path) -> Path:
    path = directory / "fake-engram"
    path.write_text(FAKE_BINARY.format(python=sys.executable), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def no_spawn(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Fail loudly if anything tries to launch ``engram serve``."""
    calls: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("engram serve must not be spawned here")

    monkeypatch.setattr(server_module, "_popen", refuse)
    return calls


@pytest.fixture
def local_port(monkeypatch: pytest.MonkeyPatch) -> int:
    """A free port that ``engram serve`` would bind (ENGRAM_PORT points at it)."""
    port = free_port()
    monkeypatch.setenv("ENGRAM_PORT", str(port))
    return port


@pytest.fixture
def fake_bin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """The fake binary as ENGRAM_BIN, reporting INSTANCE_ID; reaps any server it started."""
    binary = write_fake_binary(tmp_path)
    marker = tmp_path / "serve-marker.json"
    monkeypatch.setenv("ENGRAM_BIN", str(binary))
    monkeypatch.setenv("FAKE_ENGRAM_ID", INSTANCE_ID)
    monkeypatch.setenv("FAKE_ENGRAM_MARKER", str(marker))
    try:
        yield binary
    finally:
        # Test teardown kills the fake it launched; the provider itself never does.
        if marker.exists():
            try:
                os.kill(json.loads(marker.read_text())["pid"], signal.SIGTERM)
            except (OSError, ValueError, KeyError):
                pass


def health_payload(instance_id: str = INSTANCE_ID, version: str = "2.0.0") -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "ok", "service": "engram", "version": version}
    if instance_id:
        payload["instance_id"] = instance_id
    return payload


# -- URL match ----------------------------------------------------------------


def test_expected_serve_port_follows_engram_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENGRAM_PORT", raising=False)
    assert expected_serve_port() == 7437
    monkeypatch.setenv("ENGRAM_PORT", "8123")
    assert expected_serve_port() == 8123
    for bad in ("", "  ", "abc", "0", "70000", "-5"):
        monkeypatch.setenv("ENGRAM_PORT", bad)
        assert expected_serve_port() == 7437, bad


@pytest.mark.parametrize(
    ("url", "spawnable"),
    [
        ("http://127.0.0.1:7437", True),
        ("http://localhost:7437", True),
        ("http://127.0.0.1:7438", False),
        ("http://127.0.0.1", False),
        ("https://127.0.0.1:7437", False),
        ("http://[::1]:7437", False),
        ("http://engram.example:7437", False),
        ("http://192.168.1.10:7437", False),
        ("not a url", False),
        ("http://127.0.0.1:7437/", True),
        ("http://127.0.0.1:7437/engram", False),
        ("http://localhost:7437/api/v1", False),
        ("http://127.0.0.1:7437?x=1", False),
    ],
)
def test_only_the_default_loopback_serve_address_is_spawnable(
    monkeypatch: pytest.MonkeyPatch, url: str, spawnable: bool
) -> None:
    monkeypatch.delenv("ENGRAM_PORT", raising=False)
    assert is_spawnable_url(url) is spawnable


# -- skip paths ---------------------------------------------------------------


def test_engram_url_in_the_environment_is_external_and_never_spawns(
    monkeypatch: pytest.MonkeyPatch, local_port: int, tmp_path: Path, no_spawn: list[Any]
) -> None:
    url = f"http://127.0.0.1:{local_port}"
    monkeypatch.setenv("ENGRAM_URL", url)

    status = ensure_server(make_config(url), str(tmp_path))

    assert status.status == EXTERNAL
    assert "ENGRAM_URL" in status.reason
    assert no_spawn == []


def test_autostart_false_is_disabled_and_never_spawns(
    local_port: int, tmp_path: Path, no_spawn: list[Any]
) -> None:
    status = ensure_server(
        make_config(f"http://127.0.0.1:{local_port}", autostart=False), str(tmp_path)
    )

    assert status.status == DISABLED
    assert "autostart" in status.reason
    assert no_spawn == []


@pytest.mark.parametrize(
    "url", ["http://engram.example:{port}", "http://127.0.0.1:{other}", "http://10.0.0.5:{port}"]
)
def test_non_loopback_or_other_port_is_external_and_never_spawns(
    local_port: int, tmp_path: Path, no_spawn: list[Any], url: str
) -> None:
    configured = url.format(port=local_port, other=local_port + 1)

    status = ensure_server(make_config(configured), str(tmp_path))

    assert status.status == EXTERNAL
    assert "engram serve" in status.reason
    assert no_spawn == []


# -- healthy server -----------------------------------------------------------


def test_healthy_server_with_matching_identity_is_adopted_without_spawning(
    local_port: int, fake_bin: Path, tmp_path: Path, no_spawn: list[Any]
) -> None:
    with FakeEngram(port=local_port) as fake:
        fake.route("GET", "/health", (200, health_payload(INSTANCE_ID)))

        status = ensure_server(make_config(fake.base_url), str(tmp_path))

    assert status.status == ADOPTED
    assert no_spawn == []


def test_foreign_instance_is_reported_never_spawned_and_never_killed(
    local_port: int, fake_bin: Path, tmp_path: Path, no_spawn: list[Any]
) -> None:
    with FakeEngram(port=local_port) as fake:
        fake.route("GET", "/health", (200, health_payload(OTHER_ID)))

        status = ensure_server(make_config(fake.base_url), str(tmp_path))

        # Still alive and answering: nothing was terminated.
        assert EngramClient(make_config(fake.base_url)).health() is not None

    assert status.status == FOREIGN
    assert "instance" in status.reason
    assert no_spawn == []


@pytest.mark.parametrize(
    ("version", "phrase"), [("1.9.0", "predates instance identity"), ("2.0.0", "did not report")]
)
def test_server_without_identity_is_used_but_never_managed(
    local_port: int,
    fake_bin: Path,
    tmp_path: Path,
    no_spawn: list[Any],
    version: str,
    phrase: str,
) -> None:
    with FakeEngram(port=local_port) as fake:
        fake.route("GET", "/health", (200, health_payload("", version=version)))

        status = ensure_server(make_config(fake.base_url), str(tmp_path))

    # Deliberate divergence from gentle-engram: still used, never managed.
    assert status.status == UNAVAILABLE
    assert not status.managed
    assert phrase in status.reason
    assert "Nothing is terminated" in status.reason
    assert no_spawn == []


def test_healthy_server_but_missing_binary_is_unavailable_without_raising(
    local_port: int, tmp_path: Path, no_spawn: list[Any]
) -> None:
    with FakeEngram(port=local_port) as fake:
        fake.route("GET", "/health", (200, health_payload(INSTANCE_ID)))

        status = ensure_server(make_config(fake.base_url), str(tmp_path))

    assert status.status == UNAVAILABLE
    assert "install Engram or set ENGRAM_BIN" in status.reason
    assert no_spawn == []


# -- server down --------------------------------------------------------------


def test_down_server_is_started_detached_with_exact_args(
    local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    url = f"http://127.0.0.1:{local_port}"
    status = ensure_server(make_config(url), str(tmp_path))

    assert status.status == STARTED, status.reason
    marker = json.loads((tmp_path / "serve-marker.json").read_text())
    assert marker["argv"][1:] == ["serve"]
    assert Path(marker["argv"][0]).name == fake_bin.name
    # start_new_session: the child leads its own session, detached from ours.
    assert marker["sid"] == marker["pid"]
    assert marker["sid"] != os.getsid(0)
    assert (tmp_path / "recall" / "engram-serve.log").exists()
    health = EngramClient(make_config(url)).health()
    assert health is not None and health["instance_id"] == INSTANCE_ID


def test_spawn_uses_exact_args_and_detached_popen_options(
    monkeypatch: pytest.MonkeyPatch, local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    seen: list[tuple[Any, dict[str, Any]]] = []
    started: list[FakeEngram] = []

    waited: list[str] = []

    class FakeChild:
        pid = 4242

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            waited.append(threading.current_thread().name)
            return 0

        def kill(self) -> None:  # pragma: no cover - asserted never called
            raise AssertionError("the spawned server must never be killed")

        terminate = kill

    def fake_popen(args: Any, **kwargs: Any) -> FakeChild:
        seen.append((args, kwargs))
        fake = FakeEngram(port=local_port)
        fake.route("GET", "/health", (200, health_payload(INSTANCE_ID)))
        started.append(fake)
        return FakeChild()

    monkeypatch.setattr(server_module, "_popen", fake_popen)
    try:
        status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))
    finally:
        for fake in started:
            fake.stop()

    assert status.status == STARTED
    assert len(seen) == 1
    args, kwargs = seen[0]
    assert args == [str(fake_bin), "serve"]
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is not None and kwargs["stdout"] is not subprocess.PIPE
    assert kwargs["stderr"] is not subprocess.PIPE
    # Never inside a repository: the child runs from the recall state dir.
    assert Path(kwargs["cwd"]).resolve() == (tmp_path / "recall").resolve()
    # Reaped on a daemon thread, never on the caller's thread.
    assert len(waited) == 1
    assert waited[0] != threading.current_thread().name


def test_spawn_enables_cloud_autosync_without_touching_our_environment(
    monkeypatch: pytest.MonkeyPatch, local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    # gentle-engram 0.2.0 (2f1e545) spawns `engram serve` with
    # ENGRAM_CLOUD_AUTOSYNC=1 on top of the inherited environment.
    monkeypatch.delenv("ENGRAM_CLOUD_AUTOSYNC", raising=False)
    monkeypatch.setenv("RECALL_TEST_MARKER", "inherited")
    seen: list[dict[str, Any]] = []
    started: list[FakeEngram] = []

    class FakeChild:
        pid = 4243

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(args: Any, **kwargs: Any) -> FakeChild:
        seen.append(kwargs)
        fake = FakeEngram(port=local_port)
        fake.route("GET", "/health", (200, health_payload(INSTANCE_ID)))
        started.append(fake)
        return FakeChild()

    monkeypatch.setattr(server_module, "_popen", fake_popen)
    try:
        status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))
    finally:
        for fake in started:
            fake.stop()

    assert status.status == STARTED
    env = seen[0]["env"]
    assert env["ENGRAM_CLOUD_AUTOSYNC"] == "1"
    # A copy of ours: ENGRAM_PORT and data-dir settings must still reach the child.
    assert env["RECALL_TEST_MARKER"] == "inherited"
    assert env["ENGRAM_PORT"] == os.environ["ENGRAM_PORT"]
    assert env is not os.environ
    assert "ENGRAM_CLOUD_AUTOSYNC" not in os.environ


def test_missing_binary_is_unavailable_with_a_clear_reason(
    local_port: int, tmp_path: Path, no_spawn: list[Any]
) -> None:
    status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))

    assert status.status == UNAVAILABLE
    assert "install Engram or set ENGRAM_BIN" in status.reason
    assert no_spawn == []


def test_binary_whose_instance_id_fails_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, local_port: int, fake_bin: Path, tmp_path: Path, no_spawn
) -> None:
    monkeypatch.setenv("FAKE_ENGRAM_ID", "not-an-id")

    status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))

    assert status.status == UNAVAILABLE
    assert "instance id" in status.reason
    assert no_spawn == []


def test_child_that_never_answers_is_unavailable_and_never_killed(
    monkeypatch: pytest.MonkeyPatch, local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(server_module, "STARTUP_DEADLINE_S", 0.4)

    class SilentChild:
        pid = 4343

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:  # pragma: no cover - asserted never called
            raise AssertionError("the spawned server must never be killed")

        terminate = kill

    monkeypatch.setattr(server_module, "_popen", lambda *a, **k: SilentChild())
    started = time.monotonic()

    status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))

    assert status.status == UNAVAILABLE
    assert "did not answer" in status.reason
    assert time.monotonic() - started < 3.0


def test_lock_held_by_another_process_never_spawns_and_polls_for_health(
    monkeypatch: pytest.MonkeyPatch,
    local_port: int,
    fake_bin: Path,
    tmp_path: Path,
    no_spawn: list[Any],
) -> None:
    fcntl = pytest.importorskip("fcntl")
    lock_path = tmp_path / "recall" / "serve.lock"
    lock_path.parent.mkdir(parents=True)
    holder = open(lock_path, "a+")  # noqa: SIM115 - held across the test on purpose
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    later: list[FakeEngram] = []

    def bring_up() -> None:
        time.sleep(0.3)
        fake = FakeEngram(port=local_port)
        fake.route("GET", "/health", (200, health_payload(INSTANCE_ID)))
        later.append(fake)

    thread = threading.Thread(target=bring_up)
    thread.start()
    try:
        status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))
    finally:
        thread.join()
        for fake in later:
            fake.stop()
        holder.close()

    assert status.status == ADOPTED
    assert no_spawn == []


def test_lock_held_and_server_never_comes_up_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    local_port: int,
    fake_bin: Path,
    tmp_path: Path,
    no_spawn: list[Any],
) -> None:
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(server_module, "STARTUP_DEADLINE_S", 0.4)
    lock_path = tmp_path / "recall" / "serve.lock"
    lock_path.parent.mkdir(parents=True)
    with open(lock_path, "a+") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))

    assert status.status == UNAVAILABLE
    assert "another" in status.reason
    assert no_spawn == []


def test_no_fcntl_skips_the_lock_gracefully(
    monkeypatch: pytest.MonkeyPatch, local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(server_module, "fcntl", None)

    status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))

    assert status.status == STARTED, status.reason


def test_ensure_server_never_raises(
    monkeypatch: pytest.MonkeyPatch, local_port: int, tmp_path: Path
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(server_module, "_probe", boom)

    status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))

    assert status.status == UNAVAILABLE
    assert "RuntimeError" in status.reason


# -- persisted state ----------------------------------------------------------


def test_server_state_round_trips_and_tolerates_corruption(tmp_path: Path) -> None:
    assert read_server_state(str(tmp_path)) is None
    write_server_state(str(tmp_path), ServerStatus(STARTED, "spawned engram serve", "http://x"))

    state = read_server_state(str(tmp_path))
    assert state is not None
    assert (state.status, state.reason) == (STARTED, "spawned engram serve")

    (tmp_path / "recall" / "server.json").write_text("{not json", encoding="utf-8")
    assert read_server_state(str(tmp_path)) is None
    (tmp_path / "recall" / "server.json").write_text('{"status": "bogus"}', encoding="utf-8")
    assert read_server_state(str(tmp_path)) is None


# -- status line --------------------------------------------------------------


@pytest.mark.parametrize(
    ("server_ok", "status", "reason", "expected"),
    [
        (True, STARTED, "", "u (reachable, started by recall)"),
        (True, ADOPTED, "", "u (reachable, adopted)"),
        (True, FOREIGN, "x", "u (reachable, foreign instance — not used: x)"),
        (True, EXTERNAL, "x", "u (reachable, externally managed)"),
        (True, DISABLED, "x", "u (reachable, autostart off)"),
        (False, UNAVAILABLE, "install Engram", "u (not running — install Engram)"),
        (False, STARTED, "", "u (unreachable)"),
        (True, "", "", "u (reachable)"),
        (False, "", "", "u (unreachable)"),
    ],
)
def test_status_line_says_how_the_server_was_obtained(
    server_ok: bool, status: str, reason: str, expected: str
) -> None:
    text = render_status_line(
        "p", server_ok=server_ok, url="u", server_origin=status, server_reason=reason
    )

    assert f"  server:       {expected}" in text.splitlines()


# -- provider wiring ----------------------------------------------------------


@pytest.fixture
def recorded_ensure(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ensure_server with a recorder that reports ``adopted``."""
    calls: list[dict[str, Any]] = []

    def fake(config: Any, hermes_home: str = "", **kwargs: Any) -> ServerStatus:
        calls.append({"home": hermes_home, "at": time.monotonic(), **kwargs})
        return ServerStatus(ADOPTED, "matching instance", config.url)

    monkeypatch.setattr(server_module, "ensure_server", fake)
    return calls


def test_primary_initialize_ensures_the_server_before_registering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_engram: FakeEngram
) -> None:
    order: list[str] = []

    def fake(config: Any, hermes_home: str = "", **kwargs: Any) -> ServerStatus:
        order.append("ensure")
        return ServerStatus(ADOPTED, "", config.url)

    monkeypatch.setattr(server_module, "ensure_server", fake)
    fake_engram.route("GET", "/project/current", (200, {"project": "p"}))

    def register(_request: Any) -> tuple[int, Any]:
        order.append("register")
        return (201, {"id": "s", "status": "created"})

    fake_engram.route("POST", "/sessions", register)
    provider = RecallProvider(config=make_config(fake_engram.base_url))
    provider.initialize("s", hermes_home=str(tmp_path), platform="cli")
    try:
        assert provider._drain_for_test()
    finally:
        provider.shutdown()

    assert order == ["ensure", "register"]
    state = read_server_state(str(tmp_path))
    assert state is not None and state.status == ADOPTED


@pytest.mark.parametrize("context", ["subagent", "cron", "flush"])
def test_non_primary_initialize_never_ensures_the_server(
    tmp_path: Path, recorded_ensure: list[dict[str, Any]], context: str
) -> None:
    provider = RecallProvider(config=make_config(f"http://127.0.0.1:{free_port()}"))
    provider.initialize("s", hermes_home=str(tmp_path), agent_context=context)
    try:
        assert provider._drain_for_test()
    finally:
        provider.shutdown()

    assert recorded_ensure == []
    assert read_server_state(str(tmp_path)) is None


def test_disabled_provider_never_ensures_the_server(
    tmp_path: Path, recorded_ensure: list[dict[str, Any]]
) -> None:
    provider = RecallProvider(config=make_config("http://127.0.0.1:1", enabled=False))
    provider.initialize("s", hermes_home=str(tmp_path))
    provider.shutdown()

    assert recorded_ensure == []


def test_initialize_does_not_block_on_a_slow_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = threading.Event()
    entered = threading.Event()

    def slow(config: Any, hermes_home: str = "", **kwargs: Any) -> ServerStatus:
        entered.set()
        release.wait(5)
        return ServerStatus(STARTED, "", config.url)

    monkeypatch.setattr(server_module, "ensure_server", slow)
    provider = RecallProvider(config=make_config(f"http://127.0.0.1:{free_port()}"))
    try:
        started = time.monotonic()
        provider.initialize("s", hermes_home=str(tmp_path))
        elapsed = time.monotonic() - started
        assert entered.wait(2), "ensure_server should run on the worker"
        assert elapsed < 0.5
        # The first turn's recall path is not held up by the start either.
        prefetch_started = time.monotonic()
        provider.prefetch("explain how the provider recalls memory")
        assert time.monotonic() - prefetch_started < 3.0
    finally:
        release.set()
        assert provider._drain_for_test(5)
        provider.shutdown()


def test_restart_is_attempted_at_most_once_per_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake(config: Any, hermes_home: str = "", **kwargs: Any) -> ServerStatus:
        calls.append("ensure")
        return ServerStatus(ADOPTED, "", config.url)

    monkeypatch.setattr(server_module, "ensure_server", fake)
    dead = f"http://127.0.0.1:{free_port()}"
    provider = RecallProvider(config=make_config(dead, project_override="p"))
    provider.initialize("s", hermes_home=str(tmp_path))
    try:
        assert provider._drain_for_test()
        # Registration is refused too, which may already spend the one restart.
        assert len(calls) <= 2
        for turn in range(4):
            provider.on_turn_start(turn, f"please remember the important thing number {turn}")
            assert provider._drain_for_test()
            provider.prefetch(f"what do we know about topic number {turn}")
            assert provider._drain_for_test()
    finally:
        provider.shutdown()

    assert calls == ["ensure", "ensure"]


def test_no_restart_for_a_server_recall_does_not_manage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake(config: Any, hermes_home: str = "", **kwargs: Any) -> ServerStatus:
        calls.append("ensure")
        return ServerStatus(FOREIGN, "other instance", config.url)

    monkeypatch.setattr(server_module, "ensure_server", fake)
    provider = RecallProvider(
        config=make_config(f"http://127.0.0.1:{free_port()}", project_override="p")
    )
    provider.initialize("s", hermes_home=str(tmp_path))
    try:
        assert provider._drain_for_test()
        provider.prefetch("what do we know about the refused server")
        assert provider._drain_for_test()
    finally:
        provider.shutdown()

    assert calls == ["ensure"]


def test_shutdown_never_stops_the_started_server(
    local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    url = f"http://127.0.0.1:{local_port}"
    provider = RecallProvider(config=make_config(url))
    provider.initialize("s", hermes_home=str(tmp_path), platform="cli")
    assert provider._drain_for_test(10)
    assert provider._server_status is not None
    assert provider._server_status.status == STARTED

    provider.shutdown()

    time.sleep(0.2)
    assert EngramClient(make_config(url)).health() is not None
    text = provider.recall_command("")
    assert f"server:       {url} (reachable, started by recall)" in text


def test_cli_status_reads_the_recorded_server_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_engram: FakeEngram
) -> None:
    import cli

    fake_engram.route("GET", "/health", (200, health_payload(INSTANCE_ID)))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ENGRAM_URL", fake_engram.base_url)
    write_server_state(str(tmp_path), ServerStatus(STARTED, "", fake_engram.base_url))

    text = cli.status_text()

    assert f"server:       {fake_engram.base_url} (reachable, started by recall)" in text


def test_recall_command_reports_why_the_server_is_not_running(tmp_path: Path) -> None:
    provider = RecallProvider(config=make_config(f"http://127.0.0.1:{free_port()}"))
    provider._hermes_home = str(tmp_path)
    provider_module._set_active_provider(provider)
    provider._server_status = ServerStatus(
        UNAVAILABLE, "Engram binary not found; install Engram or set ENGRAM_BIN", ""
    )

    text = provider.recall_command("")

    assert "(not running — Engram binary not found; install Engram or set ENGRAM_BIN)" in text


# -- verifier follow-ups --------------------------------------------------------


class _QuietChild:
    """A spawned child that is alive (``poll`` is None) and reaps instantly."""

    pid = 4444

    def __init__(self, code: int | None = None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code

    def wait(self, timeout: float | None = None) -> int:
        return 0 if self.code is None else self.code

    def kill(self) -> None:  # pragma: no cover - asserted never called
        raise AssertionError("the spawned server must never be killed")

    terminate = kill


def test_shutdown_before_the_spawn_never_spawns(
    local_port: int, fake_bin: Path, tmp_path: Path, no_spawn: list[Any]
) -> None:
    status = ensure_server(
        make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path), should_stop=lambda: True
    )

    assert status.status == UNAVAILABLE
    assert "shut down" in status.reason
    assert no_spawn == []


def test_a_child_that_exits_later_is_reaped_not_left_a_zombie(
    monkeypatch: pytest.MonkeyPatch, local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_ENGRAM_LIFETIME", "0.6")

    status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))
    assert status.status == STARTED, status.reason
    pid = json.loads((tmp_path / "serve-marker.json").read_text())["pid"]

    def gone() -> bool:
        # kill(pid, 0) still succeeds for a zombie; only a reaped pid is gone.
        # (No subprocess here: subprocess._cleanup() would reap it for us.)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not gone():
        time.sleep(0.05)
    assert gone()


def test_spawned_child_runs_from_the_recall_state_dir(
    local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))

    assert status.status == STARTED, status.reason
    marker = json.loads((tmp_path / "serve-marker.json").read_text())
    assert Path(marker["cwd"]).resolve() == (tmp_path / "recall").resolve()


def test_an_oversized_serve_log_is_rotated_before_appending(
    monkeypatch: pytest.MonkeyPatch, local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    log = tmp_path / "recall" / "engram-serve.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"x" * (server_module.LOG_MAX_BYTES + 10))
    started: list[FakeEngram] = []

    def fake_popen(args: Any, **kwargs: Any) -> _QuietChild:
        fake = FakeEngram(port=local_port)
        fake.route("GET", "/health", (200, health_payload(INSTANCE_ID)))
        started.append(fake)
        return _QuietChild()

    monkeypatch.setattr(server_module, "_popen", fake_popen)
    try:
        status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))
    finally:
        for fake in started:
            fake.stop()

    assert status.status == STARTED
    rotated = log.with_name("engram-serve.log.1")
    assert rotated.stat().st_size == server_module.LOG_MAX_BYTES + 10
    assert log.stat().st_size == 0


def test_a_small_serve_log_is_appended_not_rotated(
    monkeypatch: pytest.MonkeyPatch, local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    log = tmp_path / "recall" / "engram-serve.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"earlier run\n")
    started: list[FakeEngram] = []

    def fake_popen(args: Any, **kwargs: Any) -> _QuietChild:
        fake = FakeEngram(port=local_port)
        fake.route("GET", "/health", (200, health_payload(INSTANCE_ID)))
        started.append(fake)
        return _QuietChild()

    monkeypatch.setattr(server_module, "_popen", fake_popen)
    try:
        ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))
    finally:
        for fake in started:
            fake.stop()

    assert log.read_bytes() == b"earlier run\n"
    assert not log.with_name("engram-serve.log.1").exists()


def test_child_that_exited_while_a_matching_server_answers_is_adopted(
    monkeypatch: pytest.MonkeyPatch, local_port: int, fake_bin: Path, tmp_path: Path
) -> None:
    started: list[FakeEngram] = []

    def fake_popen(args: Any, **kwargs: Any) -> _QuietChild:
        # The port was taken by a matching instance; our child exits on its own.
        fake = FakeEngram(port=local_port)
        fake.route("GET", "/health", (200, health_payload(INSTANCE_ID)))
        started.append(fake)
        return _QuietChild(code=0)

    monkeypatch.setattr(server_module, "_popen", fake_popen)
    try:
        status = ensure_server(make_config(f"http://127.0.0.1:{local_port}"), str(tmp_path))
    finally:
        for fake in started:
            fake.stop()

    assert status.status == ADOPTED


# -- foreign instance: no memory traffic ---------------------------------------


def _wire_memory(fake: FakeEngram) -> None:
    fake.route("GET", "/health", (200, health_payload(OTHER_ID)))
    fake.route("GET", "/project/current", (200, {"project": "p"}))
    fake.route("GET", "/search", (200, [{"id": 7, "type": "decision", "title": "T"}]))
    fake.route("GET", "/context", (200, {"context": ""}))
    fake.route("POST", "/sessions", (201, {"id": "s", "status": "created"}))
    fake.route("POST", "/prompts", (201, {"id": 1, "status": "saved"}))
    fake.route("POST", "/observations", (201, {"id": 9, "status": "saved"}))
    fake.route("POST", "/observations/passive", (200, {"saved": 1}))


FOREIGN_REASON = "Engram server at u is a different instance than ENGRAM_BIN"


def test_a_foreign_instance_gets_no_memory_traffic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_engram: FakeEngram
) -> None:
    _wire_memory(fake_engram)
    monkeypatch.setattr(
        server_module,
        "ensure_server",
        lambda config, home="", **k: ServerStatus(FOREIGN, FOREIGN_REASON, config.url),
    )
    provider = RecallProvider(config=make_config(fake_engram.base_url))
    provider.initialize("s", hermes_home=str(tmp_path), platform="cli")
    try:
        assert provider._drain_for_test()
        long_text = "a tool result long enough to be captured passively " * 3
        assert provider.prefetch("what do we know about the foreign server") == ""
        provider.queue_prefetch("what do we know about the foreign server")
        provider.on_turn_start(1, "please remember this important prompt text")
        provider.sync_turn(
            "u",
            long_text,
            messages=[{"role": "user", "content": "u"}, {"role": "tool", "content": long_text}],
        )
        provider.on_delegation("a delegated task " * 5, "a subagent result " * 5)
        assert provider.on_pre_compress([{"role": "user", "content": long_text}]) == ""
        provider.on_session_switch("s2", reset=True)
        provider.on_session_end([])
        assert provider._drain_for_test()
        result = json.loads(provider.handle_tool_call("mem_search", {"query": "x"}))
        saved = json.loads(provider.handle_tool_call("mem_save", {"title": "T", "content": "C"}))
        assert fake_engram.requests == []

        text = provider.recall_command("")
    finally:
        provider.shutdown()

    assert "foreign instance" in result["error"]
    assert FOREIGN_REASON in result["error"]
    assert FOREIGN_REASON in saved["error"]
    assert f"(reachable, foreign instance — not used: {FOREIGN_REASON})" in text
    # Only the status probe reached the server.
    assert {r.path for r in fake_engram.requests} == {"/health"}


@pytest.mark.parametrize(
    "reason",
    [
        "Engram server at u did not report its instance identity (server 2.0.0)",
        'cannot verify the running server: Engram binary "engram" not found',
    ],
)
def test_identity_less_or_unverified_server_is_still_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_engram: FakeEngram, reason: str
) -> None:
    _wire_memory(fake_engram)
    monkeypatch.setattr(
        server_module,
        "ensure_server",
        lambda config, home="", **k: ServerStatus(UNAVAILABLE, reason, config.url),
    )
    provider = RecallProvider(config=make_config(fake_engram.base_url))
    provider.initialize("s", hermes_home=str(tmp_path), platform="cli")
    try:
        assert provider._drain_for_test()
        text = provider.prefetch("what do we know about the unverified server")
        result = json.loads(provider.handle_tool_call("mem_search", {"query": "x"}))
    finally:
        provider.shutdown()

    assert "#7" in text
    assert "error" not in result
    assert fake_engram.find("POST", "/sessions") is not None
