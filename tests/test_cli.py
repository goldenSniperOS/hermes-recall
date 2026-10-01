"""The ``/recall`` slash command, ``hermes recall`` CLI and their shared formatting."""

from __future__ import annotations

import argparse
import importlib.util
import socket
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import REPO_ROOT
from fake_engram import FakeEngram

import hermes_recall
from hermes_recall import provider as provider_module
from hermes_recall.config import RecallConfig
from hermes_recall.format import DETAIL_CONTENT_MAX_CHARS, render_observation_detail
from hermes_recall.provider import RecallProvider
from hermes_recall.state import write_state

PROJECT = "hermes-recall"

OBSERVATION = {
    "id": 761,
    "type": "decision",
    "title": "Use stdlib urllib",
    "content": "line one\nline two",
    "project": PROJECT,
    "created_at": "2026-01-02 03:04:05",
}


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


def load_root_cli() -> ModuleType:
    """Import the repository-root ``cli.py`` by path, as the loader does (flat here)."""
    spec = importlib.util.spec_from_file_location("recall_root_cli", REPO_ROOT / "cli.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeContext:
    def __init__(self) -> None:
        self.provider: Any = None
        self.commands: dict[str, dict[str, Any]] = {}

    def register_memory_provider(self, provider: Any) -> None:
        self.provider = provider

    def register_command(
        self,
        name: str,
        handler: Any,
        description: str = "",
        args_hint: str = "",
        argument_mode: str | None = None,
    ) -> None:
        self.commands[name] = {
            "handler": handler,
            "description": description,
            "args_hint": args_hint,
        }


class InspectingContext(FakeContext):
    """Mimics ``_ProviderCollector(register_skills=False)``: inspection only."""

    _register_skills = False


class BareContext:
    def __init__(self) -> None:
        self.provider: Any = None

    def register_memory_provider(self, provider: Any) -> None:
        self.provider = provider


class ExplodingContext(FakeContext):
    def register_command(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def forget_active_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "_active_provider", None)


# -- register(ctx) ------------------------------------------------------------


def test_register_adds_the_recall_slash_command() -> None:
    ctx = FakeContext()
    hermes_recall.register(ctx)

    assert isinstance(ctx.provider, RecallProvider)
    command = ctx.commands["recall"]
    assert command["handler"] == ctx.provider.recall_command
    assert command["args_hint"] == "[id]"
    assert command["description"]


def test_register_skips_the_command_when_only_inspecting() -> None:
    ctx = InspectingContext()
    hermes_recall.register(ctx)
    assert isinstance(ctx.provider, RecallProvider)
    assert ctx.commands == {}


@pytest.mark.parametrize("ctx_type", [BareContext, ExplodingContext])
def test_register_survives_a_missing_or_failing_register_command(ctx_type: type) -> None:
    ctx = ctx_type()
    hermes_recall.register(ctx)
    assert isinstance(ctx.provider, RecallProvider)


# -- /recall ------------------------------------------------------------------


def test_recall_status_shows_project_health_last_id_and_recall_count(
    fake_engram: FakeEngram, tmp_path: Path
) -> None:
    fake_engram.route("GET", "/health", (200, {"status": "ok", "service": "engram"}))
    fake_engram.route("GET", "/project/current", (200, {"project": PROJECT}))
    fake_engram.route("POST", "/sessions", (201, {"id": "s", "status": "created"}))
    fake_engram.route("GET", "/search", (200, [OBSERVATION]))
    fake_engram.route("GET", "/context", (200, {"context": ""}))
    write_state(tmp_path, last_memory_id=761, last_action="saved", project=PROJECT)
    provider = RecallProvider(config=make_config(fake_engram.base_url))
    provider.initialize("s", hermes_home=str(tmp_path), platform="cli")
    try:
        assert provider._drain_for_test()
        provider.prefetch("why does the loader skip my provider")

        text = provider.recall_command("")
    finally:
        provider.shutdown()

    assert fake_engram.find("GET", "/health") is not None
    assert f"project:      {PROJECT}" in text
    # The fake listens on a port `engram serve` would not bind, so recall records
    # the server as externally managed rather than started or adopted.
    assert f"server:       {fake_engram.base_url} (reachable, externally managed)" in text
    assert "last memory:  #761" in text
    assert "last action:  saved" in text
    assert "last recall:  1 memory" in text


def test_recall_status_with_engram_down_reports_unreachable(tmp_path: Path) -> None:
    provider = RecallProvider(config=make_config(f"http://127.0.0.1:{unused_port()}"))
    provider._hermes_home = str(tmp_path)

    text = provider.recall_command("")

    assert "(unreachable)" in text
    assert "unreachable (" in text  # the recorded reason, on the last error line
    assert "last memory:  —" in text


def test_recall_with_an_id_shows_the_observation(fake_engram: FakeEngram) -> None:
    fake_engram.route("GET", "/observations/761", (200, OBSERVATION))
    provider = RecallProvider(config=make_config(fake_engram.base_url))

    text = provider.recall_command(" 761 ")

    assert text.startswith("🧠 observation #761")
    assert "title:    Use stdlib urllib" in text
    assert "type:     decision" in text
    assert "created:  2026-01-02 03:04:05" in text
    assert text.endswith("line one\nline two")


def test_recall_with_a_bad_or_missing_id(fake_engram: FakeEngram) -> None:
    provider = RecallProvider(config=make_config(fake_engram.base_url))

    assert "not an observation id" in provider.recall_command("abc")
    assert "not available" in provider.recall_command("5")


def test_recall_command_delegates_to_the_initialized_instance(tmp_path: Path) -> None:
    registered = RecallProvider(config=make_config(f"http://127.0.0.1:{unused_port()}"))
    running = RecallProvider(config=make_config(f"http://127.0.0.1:{unused_port()}"))
    running.initialize("s", hermes_home=str(tmp_path), platform="cli", agent_context="cron")
    try:
        running._project = "running-project"
        assert "project:      running-project" in registered.recall_command("")
    finally:
        running.shutdown()


def test_observation_detail_caps_content() -> None:
    text = render_observation_detail({**OBSERVATION, "content": "z" * 5000})
    body = text.split("\n\n", 1)[1]
    assert len(body) == DETAIL_CONTENT_MAX_CHARS
    assert body.endswith("\u2026")


def test_observation_detail_without_an_observation() -> None:
    assert render_observation_detail(None, error="HTTP 404") == (
        "🧠 Recall — observation not available (HTTP 404)"
    )


# -- hermes recall (cli.py) ----------------------------------------------------


def test_root_cli_register_cli_builds_status_and_show() -> None:
    cli = load_root_cli()
    parser = argparse.ArgumentParser(prog="hermes recall")

    cli.register_cli(parser)

    args = parser.parse_args(["status"])
    assert args.recall_action == "status"
    assert args.func is cli.recall_command
    show = parser.parse_args(["show", "761"])
    assert (show.recall_action, show.id) == ("show", "761")
    # The loader resolves the handler as `<provider name>_command`.
    assert callable(cli.recall_command)


def test_cli_status_prints_the_same_status_text(
    fake_engram: FakeEngram,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_engram.route("GET", "/health", (200, {"status": "ok"}))
    monkeypatch.setenv("ENGRAM_URL", fake_engram.base_url)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    write_state(tmp_path, last_memory_id=42, last_action="updated", project=PROJECT)
    cli = load_root_cli()
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)

    assert cli.recall_command(parser.parse_args(["status"])) == 0

    out = capsys.readouterr().out
    assert out.startswith("🧠 Recall — Engram memory provider")
    assert "last memory:  #42" in out
    assert "last action:  updated" in out
    assert f"project:      {PROJECT}" in out
    assert "(reachable)" in out


def test_cli_show_prints_the_detail_view(
    fake_engram: FakeEngram,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_engram.route("GET", "/observations/761", (200, OBSERVATION))
    monkeypatch.setenv("ENGRAM_URL", fake_engram.base_url)
    cli = load_root_cli()
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)

    assert cli.recall_command(parser.parse_args(["show", "761"])) == 0
    assert capsys.readouterr().out.startswith("🧠 observation #761")


def test_cli_without_a_subcommand_shows_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ENGRAM_URL", f"http://127.0.0.1:{unused_port()}")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cli = load_root_cli()
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)

    assert cli.recall_command(parser.parse_args([])) == 0
    assert "(unreachable)" in capsys.readouterr().out
