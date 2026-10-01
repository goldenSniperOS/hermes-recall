"""Engram-compatible memory provider for Hermes Agent.

``register`` is the canonical registration entry point for both install routes:
the git-clone directory install (through the repository root ``__init__.py``,
which re-exports it) and the pip install (through the
``hermes_agent.memory_providers`` entry point ``recall = "hermes_recall:register"``).
"""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"
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

    provider = RecallProvider()
    ctx.register_memory_provider(provider)
    _register_slash_command(ctx, provider)


def _register_slash_command(ctx: Any, provider: Any) -> None:
    """Register ``/recall [id]`` when the context supports it. Never raises.

    For an exclusive provider ``ctx`` is the loader's ``_ProviderCollector``, whose
    ``__getattr__`` forwards every ``register_*`` call to a real ``PluginContext``
    (``plugins/memory/__init__.py``). The call is guarded so a host without
    ``register_command`` — or one whose registration fails — can never cost the
    provider registration that already happened above.

    Skipped when the collector says it is only *inspecting* the provider
    (``register_skills=False``: ``hermes memory status``, the setup picker), so a
    merely installed, inactive provider does not leak ``/recall`` into sessions.
    """
    try:
        if getattr(ctx, "_register_skills", True) is False:
            return
        register_command = getattr(ctx, "register_command", None)
        if not callable(register_command):
            return
        register_command(
            "recall",
            provider.recall_command,
            description="Show Engram recall status, or one memory: /recall [id]",
            args_hint="[id]",
        )
    except Exception:
        pass
