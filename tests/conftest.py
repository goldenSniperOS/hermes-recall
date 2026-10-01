"""Shared test setup: make the repository root importable for flat imports.

The Hermes memory loader imports the repository root ``__init__.py`` with the
plugin directory as the package ``__path__`` and does no ``sys.path`` work. Tests
have no loader, so they take the other route the root ``__init__.py`` supports:
the repository root on ``sys.path``, then ``import hermes_recall``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_repo_on_path() -> None:
    """Prepend the repository root to ``sys.path`` (idempotent)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_on_path()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, for tests that read shipped files."""
    return REPO_ROOT
