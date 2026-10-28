"""Shared test setup: make the repository root importable for flat imports.

The Hermes memory loader imports the repository root ``__init__.py`` with the
plugin directory as the package ``__path__`` and does no ``sys.path`` work. Tests
have no loader, so they take the other route the root ``__init__.py`` supports:
the repository root on ``sys.path``, then ``import hermes_recall``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_repo_on_path() -> None:
    """Prepend the repository root to ``sys.path`` (idempotent)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_on_path()

# Imported after the path fix-up and so below the other imports: ``fake_engram``
# lives beside this file, which is on ``sys.path`` because pytest puts a
# conftest's own directory there when it imports it.
from fake_engram import FakeEngram  # noqa: E402
from fake_mcp import FakeMcp  # noqa: E402

from hermes_recall import mcp_proxy  # noqa: E402


@pytest.fixture(autouse=True)
def _never_reach_the_real_engram(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep the auto-start path away from the user's real Engram, in every test.

    The provider only spawns or probes for auto-start when the configured URL is the
    address a plain ``engram serve`` would bind, which follows ``ENGRAM_PORT``. Port 1
    is never what a test config points at, so the default URL can never match; tests
    that exercise auto-start set ``ENGRAM_PORT`` to their own free port. ``ENGRAM_BIN``
    points at a path that does not exist, so the real binary is never run (and the
    MCP proxy falls back to the HTTP tools), and an ambient ``ENGRAM_URL`` would turn
    every auto-start test into a no-op. ``HERMES_HOME`` points at a scratch
    directory, so a provider asked for tool schemas before ``initialize`` never
    writes under the user's ``~/.hermes``.
    """
    monkeypatch.setenv("ENGRAM_PORT", "1")
    missing = tmp_path_factory.getbasetemp() / "no-such-engram-binary"
    monkeypatch.setenv("ENGRAM_BIN", str(missing))
    monkeypatch.delenv("ENGRAM_URL", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path_factory.getbasetemp() / "hermes-home"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, for tests that read shipped files."""
    return REPO_ROOT


@pytest.fixture
def fake_engram() -> Iterator[FakeEngram]:
    """A started in-process Engram stand-in, stopped on teardown.

    Function-scoped on purpose: tests assert on the recorded request log, so each
    one needs a server that has seen nothing yet.
    """
    server = FakeEngram()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(autouse=True)
def _fresh_mcp_discovery(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No MCP tool discovery leaks between tests through the in-process cache.

    The proxy's bounds shrink too, so a test that hangs the fake child waits seconds,
    not the production timeouts.
    """
    mcp_proxy.clear_discovery_cache()
    monkeypatch.setattr(mcp_proxy, "CALL_TIMEOUT_S", 2.0)
    monkeypatch.setattr(mcp_proxy, "START_TIMEOUT_S", 5.0)
    yield
    mcp_proxy.clear_discovery_cache()


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Hermes home of the test's own, also what ``HERMES_HOME`` names."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def fake_mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeMcp:
    """The fake ``engram`` MCP binary, installed as ``ENGRAM_BIN``."""
    fake = FakeMcp(tmp_path / "bin")
    monkeypatch.setenv("ENGRAM_BIN", str(fake.binary))
    return fake
