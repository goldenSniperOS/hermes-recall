"""Hermes memory provider entry point for ``hermes plugins install``.

The installer clones this repository into ``$HERMES_HOME/plugins/recall`` and the
Hermes memory loader imports this file as ``_hermes_user_memory.recall`` with the
plugin directory as the package ``__path__`` (see ``_load_provider_from_dir`` in
``plugins/memory/__init__.py``). The provider name is the **directory** name, so
that directory must be ``recall`` — the same value ``memory.provider`` takes in
config.yaml and the same value ``RecallProvider.name`` returns.

The loader does no ``sys.path`` setup, hence the relative import below: it
resolves the ``hermes_recall`` package shipped next to this file. Tests and other
flat imports put the repository root on ``sys.path`` and use the absolute import
instead.

``register(ctx)`` is re-exported from ``hermes_recall`` and calls
``ctx.register_memory_provider(...)`` with the ``MemoryProvider`` subclass
``hermes_recall.provider.RecallProvider``. Those two names also satisfy the
loader's discovery heuristic, which text-scans the first 8192 bytes of this file
for ``register_memory_provider`` or ``MemoryProvider``.
"""

from __future__ import annotations

try:  # loaded as a package by the Hermes memory provider loader
    from .hermes_recall import __version__, register
except ImportError:  # imported flat (repository root on sys.path)
    from hermes_recall import __version__, register

__all__ = ["__version__", "register"]
