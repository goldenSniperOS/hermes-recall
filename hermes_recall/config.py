"""Settings resolution for the recall provider: environment, ``memory.recall.*``, defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_URL = "http://127.0.0.1:7437"
DEFAULT_RECALL_LIMIT = 5
DEFAULT_RECALL_MAX_CHARS = 2500

# Clamp bounds. A recall block injected into every turn is a cost multiplier, so
# both knobs are bounded rather than trusted: the lower bound keeps the feature
# meaningful, the upper bound keeps a typo out of the prompt budget.
_RECALL_LIMIT_MIN = 1
_RECALL_LIMIT_MAX = 20
_RECALL_MAX_CHARS_MIN = 200
_RECALL_MAX_CHARS_MAX = 20000

# The only strings accepted for `enabled`, so a hand-edited config.yaml that says
# `enabled: "no"` (a string, not a YAML bool) still disables the provider.
_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class RecallConfig:
    """Fully resolved provider settings. Every field is normalised and in range."""

    url: str
    """Base URL of the Engram server, with no trailing slash."""

    project_override: str
    """Explicit project name, or ``""`` to let Engram detect it from the cwd."""

    recall_limit: int
    """Maximum search hits injected per turn, clamped to 1..20."""

    recall_max_chars: int
    """Character ceiling on the injected recall block, clamped to 200..20000."""

    enabled: bool
    """``False`` disables recall and capture without uninstalling the plugin."""

    auth_token: str
    """Bearer token for the Engram server, or ``""`` when unset."""


def _load_recall_settings() -> dict[str, Any]:
    """Return the ``memory.recall`` block from the Hermes config (``{}`` on any error).

    The ``hermes_cli`` import is deliberately lazy and inside the ``try``: this
    module has to import cleanly with no Hermes runtime present. That is not a
    hypothetical — it is how the test suite runs (Hermes is not a dependency of
    this plugin, so it is absent from the dev virtualenv), and it is also what
    keeps a broken or partially installed Hermes from turning settings lookup
    into an import error. Every failure mode collapses to "no config found", so
    the caller falls back to defaults.
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        memory_config = config.get("memory", {}) if isinstance(config, dict) else {}
        recall_config = memory_config.get("recall", {}) if isinstance(memory_config, dict) else {}
        return dict(recall_config) if isinstance(recall_config, dict) else {}
    except Exception:
        return {}


def _config_str(value: Any) -> str:
    """Return a stripped string for a config value, else ``""``."""
    return value.strip() if isinstance(value, str) else ""


def _normalize_url(raw: str) -> str:
    """Strip whitespace and trailing slashes; a blank result becomes the default."""
    cleaned = raw.strip().rstrip("/")
    return cleaned or DEFAULT_URL


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    """Coerce ``value`` to an int clamped to ``minimum..maximum``.

    Accepts an ``int`` or a numeric ``str``. ``None``, bools and anything
    unparseable fall back to ``default``. Bools are rejected explicitly because
    ``isinstance(True, int)`` is true in Python, and ``enabled: true`` landing in
    a numeric field should mean "I did not set this", not ``1``.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except (TypeError, ValueError):
            return default
    else:
        return default
    return max(minimum, min(maximum, parsed))


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Coerce ``value`` to a bool, falling back to ``default`` when unrecognised."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    return default


def load_config() -> RecallConfig:
    """Resolve provider settings with precedence: environment > Hermes config > default.

    Never raises: a failure anywhere in the chain degrades to the documented
    default for that field, because a settings problem must not stop Hermes from
    starting a turn.
    """
    try:
        settings = _load_recall_settings()

        env_url = os.environ.get("ENGRAM_URL", "")
        url = (
            _normalize_url(env_url)
            if env_url.strip()
            else _normalize_url(_config_str(settings.get("url")))
        )

        return RecallConfig(
            url=url,
            project_override=_config_str(settings.get("project")),
            recall_limit=_coerce_int(
                settings.get("recall_limit"),
                default=DEFAULT_RECALL_LIMIT,
                minimum=_RECALL_LIMIT_MIN,
                maximum=_RECALL_LIMIT_MAX,
            ),
            recall_max_chars=_coerce_int(
                settings.get("recall_max_chars"),
                default=DEFAULT_RECALL_MAX_CHARS,
                minimum=_RECALL_MAX_CHARS_MIN,
                maximum=_RECALL_MAX_CHARS_MAX,
            ),
            enabled=_coerce_bool(settings.get("enabled"), default=True),
            auth_token=os.environ.get("ENGRAM_HTTP_TOKEN", "").strip(),
        )
    except Exception:
        return RecallConfig(
            url=DEFAULT_URL,
            project_override="",
            recall_limit=DEFAULT_RECALL_LIMIT,
            recall_max_chars=DEFAULT_RECALL_MAX_CHARS,
            enabled=True,
            auth_token="",
        )


def resolve_hermes_home(explicit: str = "") -> str:
    """Return the Hermes home directory: ``explicit``, else ``$HERMES_HOME``, else ``~/.hermes``.

    ``explicit`` is the ``hermes_home`` kwarg Hermes passes to ``initialize`` (the
    active profile's home). The fallbacks serve callers that have no provider
    session — the ``hermes recall`` CLI — and never raise.
    """
    try:
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        env_home = os.environ.get("HERMES_HOME", "")
        if env_home.strip():
            return env_home.strip()
        return os.path.join(os.path.expanduser("~"), ".hermes")
    except Exception:
        return os.path.join("~", ".hermes")
