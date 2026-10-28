"""Settings precedence, coercion and clamping for ``hermes_recall.config``."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from hermes_recall.config import (
    DEFAULT_RECALL_LIMIT,
    DEFAULT_RECALL_MAX_CHARS,
    DEFAULT_URL,
    RecallConfig,
    load_config,
)

ENV_URL = "ENGRAM_URL"
ENV_TOKEN = "ENGRAM_HTTP_TOKEN"


def stub_hermes_config(
    monkeypatch: pytest.MonkeyPatch, recall: Any, *, raises: bool = False
) -> None:
    """Install a fake ``hermes_cli.config`` whose ``memory.recall`` block is ``recall``.

    ``config.py`` imports ``load_config_readonly`` lazily inside the function, so
    the stub only has to be in ``sys.modules`` by the time ``load_config`` runs.
    ``hermes_cli`` itself is stubbed too, because importing a submodule of a
    package that is not in ``sys.modules`` would go to the real (absent) one.
    """

    def load_config_readonly() -> Any:
        if raises:
            raise RuntimeError("config file is corrupt")
        return {"memory": {"recall": recall}}

    package = ModuleType("hermes_cli")
    config_module = ModuleType("hermes_cli.config")
    config_module.load_config_readonly = load_config_readonly  # type: ignore[attr-defined]
    package.config = config_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)


def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove both environment variables the config reads."""
    monkeypatch.delenv(ENV_URL, raising=False)
    monkeypatch.delenv(ENV_TOKEN, raising=False)
    monkeypatch.delenv("ENGRAM_PORT", raising=False)


# -- url ---------------------------------------------------------------------


def test_env_url_wins_over_config_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    monkeypatch.setenv(ENV_URL, "http://127.0.0.1:9999")
    stub_hermes_config(monkeypatch, {"url": "http://config-host:1234"})

    assert load_config().url == "http://127.0.0.1:9999"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://127.0.0.1:9999/", "http://127.0.0.1:9999"),
        ("http://127.0.0.1:9999///", "http://127.0.0.1:9999"),
        ("  http://127.0.0.1:9999/  ", "http://127.0.0.1:9999"),
    ],
)
def test_env_url_trailing_slash_and_whitespace_stripped(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    clear_env(monkeypatch)
    monkeypatch.setenv(ENV_URL, raw)
    stub_hermes_config(monkeypatch, {})

    assert load_config().url == expected


def test_config_url_trailing_slash_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"url": "http://config-host:1234/"})

    assert load_config().url == "http://config-host:1234"


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_env_url_falls_back_to_config(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    clear_env(monkeypatch)
    monkeypatch.setenv(ENV_URL, blank)
    stub_hermes_config(monkeypatch, {"url": "http://config-host:1234"})

    assert load_config().url == "http://config-host:1234"


def test_blank_env_and_blank_config_url_fall_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_env(monkeypatch)
    monkeypatch.setenv(ENV_URL, "   ")
    stub_hermes_config(monkeypatch, {"url": "   "})

    assert load_config().url == DEFAULT_URL


def test_default_url_follows_engram_port(monkeypatch: pytest.MonkeyPatch) -> None:
    # gentle-engram: `http://127.0.0.1:${ENGRAM_PORT ?? 7437}` (index.ts:24-26).
    clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_PORT", "8123")
    stub_hermes_config(monkeypatch, {})

    assert load_config().url == "http://127.0.0.1:8123"


@pytest.mark.parametrize("raw", ["", "  ", "abc", "0", "70000"])
def test_invalid_engram_port_keeps_the_7437_default(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_PORT", raw)
    stub_hermes_config(monkeypatch, {})

    assert load_config().url == DEFAULT_URL


def test_explicit_urls_win_over_engram_port(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_PORT", "8123")
    stub_hermes_config(monkeypatch, {"url": "http://config-host:1234"})
    assert load_config().url == "http://config-host:1234"

    monkeypatch.setenv(ENV_URL, "http://env-host:999")
    assert load_config().url == "http://env-host:999"


def test_config_url_used_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"url": "http://config-host:1234"})

    assert load_config().url == "http://config-host:1234"


def test_defaults_when_neither_env_nor_config_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {})

    config = load_config()
    assert config == RecallConfig(
        url=DEFAULT_URL,
        project_override="",
        recall_limit=DEFAULT_RECALL_LIMIT,
        recall_max_chars=DEFAULT_RECALL_MAX_CHARS,
        enabled=True,
        auth_token="",
        autostart=True,
    )


# -- recall_limit / recall_max_chars -----------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (7, 7),  # valid int
        ("9", 9),  # numeric string
        ("  9  ", 9),  # numeric string with whitespace
        ("garbage", DEFAULT_RECALL_LIMIT),  # unparseable
        (None, DEFAULT_RECALL_LIMIT),  # absent / explicit null
        (True, DEFAULT_RECALL_LIMIT),  # bool is not a count
        ([], DEFAULT_RECALL_LIMIT),  # wrong type entirely
        (0, 1),  # below minimum -> clamped up
        (-50, 1),  # far below minimum
        (999, 20),  # above maximum -> clamped down
    ],
)
def test_recall_limit_coercion_and_clamping(
    monkeypatch: pytest.MonkeyPatch, raw: Any, expected: int
) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"recall_limit": raw})

    assert load_config().recall_limit == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (4000, 4000),
        ("4000", 4000),
        ("nope", DEFAULT_RECALL_MAX_CHARS),
        (None, DEFAULT_RECALL_MAX_CHARS),
        (False, DEFAULT_RECALL_MAX_CHARS),
        ({"a": 1}, DEFAULT_RECALL_MAX_CHARS),
        (10, 200),  # below minimum
        (10**9, 20000),  # above maximum
    ],
)
def test_recall_max_chars_coercion_and_clamping(
    monkeypatch: pytest.MonkeyPatch, raw: Any, expected: int
) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"recall_max_chars": raw})

    assert load_config().recall_max_chars == expected


def test_recall_ints_ignore_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only `url` and `auth_token` have env overrides; the two sizing knobs are
    # config-only, so a same-named env var must not leak in.
    clear_env(monkeypatch)
    monkeypatch.setenv("RECALL_LIMIT", "19")
    monkeypatch.setenv("RECALL_MAX_CHARS", "19000")
    stub_hermes_config(monkeypatch, {})

    config = load_config()
    assert config.recall_limit == DEFAULT_RECALL_LIMIT
    assert config.recall_max_chars == DEFAULT_RECALL_MAX_CHARS


# -- enabled ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        ("on", True),
        ("off", False),
        ("TRUE", True),
        ("Off", False),
        ("  no  ", False),
        ("maybe", True),  # unrecognised -> default True
        (3, True),  # wrong type -> default True
        (None, True),  # absent -> default True
    ],
)
def test_enabled_coercion(monkeypatch: pytest.MonkeyPatch, raw: Any, expected: bool) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"enabled": raw})

    assert load_config().enabled is expected


# -- autostart ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        ("false", False),
        ("off", False),
        ("yes", True),
        ("maybe", True),  # unrecognised -> default True
        (None, True),  # absent -> default True
    ],
)
def test_autostart_coercion(monkeypatch: pytest.MonkeyPatch, raw: Any, expected: bool) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"autostart": raw} if raw is not None else {})

    assert load_config().autostart is expected


def test_autostart_is_independent_of_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"enabled": True, "autostart": False})

    config = load_config()
    assert config.enabled is True
    assert config.autostart is False


# -- tools allowlist ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ()),  # absent -> every tool
        ([], ()),
        ("", ()),
        (["mem_save", " mem_search ", "mem_save", "", 3], ("mem_save", "mem_search")),
        ("mem_save, mem_search,,mem_save", ("mem_save", "mem_search")),
        (42, ()),  # wrong type -> every tool
    ],
)
def test_tools_allowlist_coercion(monkeypatch: pytest.MonkeyPatch, raw: Any, expected: Any) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"tools": raw} if raw is not None else {})

    assert load_config().tools == expected


# -- project_override ---------------------------------------------------------


def test_project_override_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"project": "  hermes-recall  "})

    assert load_config().project_override == "hermes-recall"


@pytest.mark.parametrize("raw", [None, "", "   ", 42, ["a"]])
def test_project_override_absent_or_wrong_type_is_blank(
    monkeypatch: pytest.MonkeyPatch, raw: Any
) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"project": raw} if raw is not None else {})

    assert load_config().project_override == ""


def test_project_override_ignores_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # `project` is config-only by design: an env override would silently
    # repoint memory writes at another project.
    clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_PROJECT", "some-other-project")
    stub_hermes_config(monkeypatch, {})

    assert load_config().project_override == ""


# -- auth_token ---------------------------------------------------------------


def test_auth_token_from_env_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    monkeypatch.setenv(ENV_TOKEN, "  s3cret-token  ")
    stub_hermes_config(monkeypatch, {})

    assert load_config().auth_token == "s3cret-token"


def test_auth_token_blank_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {})

    assert load_config().auth_token == ""


def test_auth_token_is_not_read_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # A secret belongs in the environment, not in a config.yaml the Dashboard
    # writes and the user commits.
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"auth_token": "from-config"})

    assert load_config().auth_token == ""


# -- defensive behaviour ------------------------------------------------------


def test_raising_hermes_config_yields_defaults_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, {"url": "http://never-read:1"}, raises=True)

    config = load_config()
    assert config.url == DEFAULT_URL
    assert config.recall_limit == DEFAULT_RECALL_LIMIT
    assert config.recall_max_chars == DEFAULT_RECALL_MAX_CHARS
    assert config.enabled is True
    assert config.project_override == ""


@pytest.mark.parametrize("shape", ["not-a-dict", 42, ["recall"], None])
def test_non_dict_config_shapes_yield_defaults(monkeypatch: pytest.MonkeyPatch, shape: Any) -> None:
    clear_env(monkeypatch)
    stub_hermes_config(monkeypatch, shape)

    assert load_config().url == DEFAULT_URL


def test_non_dict_top_level_config_yields_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_env(monkeypatch)
    module = ModuleType("hermes_cli.config")
    module.load_config_readonly = lambda: ["not", "a", "dict"]  # type: ignore[attr-defined]
    package = ModuleType("hermes_cli")
    package.config = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", module)

    assert load_config().url == DEFAULT_URL


def test_missing_load_config_readonly_symbol_yields_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Hermes version that renamed or dropped the helper must not break us.
    clear_env(monkeypatch)
    module = ModuleType("hermes_cli.config")
    package = ModuleType("hermes_cli")
    package.config = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", module)

    assert load_config().url == DEFAULT_URL


def test_genuine_no_hermes_path_yields_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real no-Hermes path, not a stub of it.

    Hermes genuinely is not installed in this plugin's dev virtualenv (its
    runtime dependencies are empty), so with nothing in ``sys.modules`` the lazy
    ``from hermes_cli.config import ...`` raises a real ``ModuleNotFoundError``
    and must be swallowed. The assertion below pins that: ``hermes_cli`` is
    absent for real, and ``load_config()`` still returns a usable config.
    """
    clear_env(monkeypatch)
    monkeypatch.delitem(sys.modules, "hermes_cli", raising=False)
    monkeypatch.delitem(sys.modules, "hermes_cli.config", raising=False)

    with pytest.raises(ModuleNotFoundError):
        import hermes_cli.config  # noqa: F401

    config = load_config()
    assert config.url == DEFAULT_URL
    assert config.enabled is True
    assert config.recall_limit == DEFAULT_RECALL_LIMIT


def test_recall_config_is_frozen() -> None:
    # Frozen on purpose: the config is read on every turn and shared with the
    # client, so nothing downstream may quietly repoint the server URL.
    config = load_config()
    with pytest.raises(FrozenInstanceError):
        config.url = "http://mutated:1"  # type: ignore[misc]


def test_load_config_never_raises_even_with_a_hostile_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-dict `os.environ` stand-in is the crudest way to break the lookups;
    # the outer guard must still produce a config.
    stub_hermes_config(monkeypatch, {})
    monkeypatch.setattr("hermes_recall.config.os", SimpleNamespace(), raising=True)

    assert load_config().url == DEFAULT_URL
