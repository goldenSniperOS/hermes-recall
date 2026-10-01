"""Engram-compatible memory provider for Hermes Agent.

``register`` is the canonical registration entry point for both install routes:
the git-clone directory install (through the repository root ``__init__.py``,
which re-exports it) and the pip install (through the
``hermes_agent.memory_providers`` entry point ``recall = "hermes_recall:register"``).
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = ["__version__", "register"]


def register(ctx: Any) -> None:
    """Hand a ``RecallProvider`` instance to the Hermes memory loader.

    ``RecallProvider`` is imported here rather than at module scope on purpose:
    it subclasses Hermes' ``MemoryProvider`` ABC, so importing it eagerly would
    pull the Hermes runtime into every plain ``import hermes_recall`` — including
    the packaging tests, which must stay runtime-free. An ``ImportError`` raised
    here propagates to the loader, which logs it; nothing is swallowed.
    """
    from .provider import RecallProvider

    ctx.register_memory_provider(RecallProvider())
