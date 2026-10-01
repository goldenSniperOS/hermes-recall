"""Hermes ``MemoryProvider`` ABC compatibility shim.

:class:`~hermes_recall.provider.RecallProvider` must subclass the *real* Hermes
``agent.memory_provider.MemoryProvider`` at runtime — the host's loader type-checks
and introspects it. But Hermes is not a dependency of this plugin, so it is absent
from the dev virtualenv and from CI. This module is the single seam that reconciles
those two facts: it imports the real ABC when Hermes is importable, and otherwise
defines a faithful, minimal stand-in.

The fallback exists **only** so the package stays importable and testable without
the Hermes runtime. Inside Hermes the real ABC always wins — the ``try`` branch is
the production path, the ``except`` branch is the test/CI path.

Drift between the fallback and upstream is not left to vigilance:
``tests/test_hermes_compat.py`` loads the upstream module from ``.refs/`` and
asserts the regex, the ``RecallStatus`` dataclass shape, the glyph and the hook
surface all still agree.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = [
    "HERMES_AVAILABLE",
    "INDICATOR_GLYPH",
    "TRIVIAL_PROMPT_RE",
    "MemoryProvider",
    "RecallStatus",
    "is_trivial_prompt",
]

try:
    from agent.memory_provider import (  # type: ignore[import-not-found]
        INDICATOR_GLYPH,
        TRIVIAL_PROMPT_RE,
        MemoryProvider,
        RecallStatus,
        is_trivial_prompt,
    )

    HERMES_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the whole test suite
    HERMES_AVAILABLE = False

    # Default glyph for Hermes' deterministic memory indicators.
    INDICATOR_GLYPH = "🧠"

    @dataclass(frozen=True)
    class RecallStatus:  # type: ignore[no-redef]
        """What the provider's most recent prefetch injected, for the host's indicator.

        Field names, order and the ``glyph`` default mirror upstream exactly:
        ``MemoryManager.describe_recall`` renders
        ``f"{glyph} {provider_label} — {detail}"``, and ``count <= 0`` means
        "content injected, no discrete count".
        """

        provider_label: str
        count: int
        glyph: str = INDICATOR_GLYPH

    # Source of truth: ``TRIVIAL_PROMPT_RE`` in Hermes'
    # ``agent/memory_provider.py``. Copied verbatim — the alternation is anchored
    # and may only be followed by whitespace or punctuation, so words that merely
    # START with a trivial word ("k8s", "yolo", "note", "hindsight") do NOT match,
    # while trailing-punctuation variants ("hi!", "thanks :)", "done???") do.
    # Only the line layout differs (this repo caps lines at 100 chars); the
    # compiled pattern string and flags are byte-identical, and
    # ``tests/test_hermes_compat.py`` asserts exactly that against the upstream
    # file so the copy cannot silently drift.
    _TRIVIAL_PROMPT_PATTERN = (
        r"^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|"
        r"hi|hey|hello|yo|sup|"
        r"continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)"
        r'[\s!?.:;,"'
        "'"
        r"~\u2018\u2019\u201c\u201d\u2014\u2013\u2026"
        r"()\[\]{}<>*&^%$#@!+=`\u00a0]*$"
    )

    TRIVIAL_PROMPT_RE = re.compile(_TRIVIAL_PROMPT_PATTERN, re.IGNORECASE)

    # Source of truth: ``is_trivial_prompt`` in Hermes'
    # ``agent/memory_provider.py``. Body copied verbatim; drift is guarded by
    # ``tests/test_hermes_compat.py``, which compares this function's verdict
    # against the upstream one across a table of inputs.
    def is_trivial_prompt(text: str | None) -> bool:  # type: ignore[no-redef]
        """Return True if a user prompt is too trivial to warrant memory recall.

        Empty/whitespace-only input, slash commands, and bare greetings or
        acknowledgements (with optional trailing punctuation) all count as
        trivial.
        """
        if not text:
            return True
        stripped = text.strip()
        if not stripped:
            return True
        if stripped.startswith("/"):
            return True
        return bool(TRIVIAL_PROMPT_RE.match(stripped))

    class MemoryProvider(ABC):  # type: ignore[no-redef]
        """Minimal stand-in for Hermes' memory provider ABC.

        Carries the same abstract members and the same optional hooks (with the
        same signatures and no-op/empty defaults) that
        :class:`~hermes_recall.provider.RecallProvider` relies on. Everything
        upstream offers beyond that surface is deliberately absent: this is a test
        scaffold, not a reimplementation.
        """

        # Version 1 is the implicit historical contract: best-effort
        # ``on_pre_compress`` with the raw message list. Upstream's module-level
        # ``PRE_COMPRESS_CHECKPOINT_API_VERSION`` is 2, but the class default is 1
        # and this provider does not opt in.
        pre_compress_checkpoint_api_version = 1

        @property
        @abstractmethod
        def name(self) -> str:
            """Short identifier for this provider."""

        # -- Core lifecycle --------------------------------------------------

        @abstractmethod
        def is_available(self) -> bool:
            """Return True when configured and ready. Must not touch the network."""

        @abstractmethod
        def initialize(self, session_id: str, **kwargs: Any) -> None:
            """Initialize for a session."""

        def unavailable_reason(self) -> str:
            """Actionable reason this provider reports unavailable."""
            return ""

        def system_prompt_block(self) -> str:
            """Static text to include in the system prompt."""
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            """Recall relevant context for the upcoming turn."""
            return ""

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
            """Queue a background recall for the NEXT turn."""

        def recall_status(self) -> RecallStatus | None:
            """Describe what the most recent prefetch injected, for the UI."""
            return None

        def sync_turn(
            self,
            user_content: str,
            assistant_content: str,
            *,
            session_id: str = "",
            messages: list[dict[str, Any]] | None = None,
        ) -> None:
            """Persist a completed turn to the backend."""

        @abstractmethod
        def get_tool_schemas(self) -> list[dict[str, Any]]:
            """Return tool schemas this provider exposes."""

        def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
            """Handle a tool call for one of this provider's tools."""
            raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

        def shutdown(self) -> None:
            """Clean shutdown — flush queues, close connections."""

        # -- Optional hooks --------------------------------------------------

        def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
            """Called at the start of each turn with the user message."""

        def on_session_end(self, messages: list[dict[str, Any]]) -> None:
            """Called when a session ends (explicit exit or timeout)."""

        def on_session_switch(
            self,
            new_session_id: str,
            *,
            parent_session_id: str = "",
            reset: bool = False,
            rewound: bool = False,
            **kwargs: Any,
        ) -> None:
            """Called when the agent switches session_id mid-process."""

        def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
            """Called before context compression discards old messages."""
            return ""

        def on_delegation(
            self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
        ) -> None:
            """Called on the PARENT agent when a subagent completes."""
