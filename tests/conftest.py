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
