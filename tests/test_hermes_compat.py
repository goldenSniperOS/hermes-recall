"""Drift detector for the Hermes ABC fallback in ``hermes_recall._hermes``.

The fallback is a hand copy of pieces of Hermes' ``agent/memory_provider.py``. A
copy rots silently, so this module loads the upstream file directly and asserts the
copy still agrees with it: the trivial-prompt regex, the classifier's verdicts, the
``RecallStatus`` dataclass shape, the indicator glyph, and the hook surface
``RecallProvider`` overrides.

The upstream file lives in ``.refs/``, which is gitignored and absent in CI. Every
test here skips when it is missing — that is the expected CI case, not a failure.
Locally, with ``.refs/`` present, they all run.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from hermes_recall import _hermes
from hermes_recall.provider import RecallProvider

UPSTREAM_RELATIVE = Path(".refs/hermes/agent/memory_provider.py")

# Inputs chosen to pin the regex's two sharp edges: words that merely START with a
# trivial word must NOT be trivial ("k8s", "yolo", "note this down", "hindsight"),
# while trailing-punctuation variants of real acknowledgements must be.
TRIVIAL_TABLE = [
    "",
    "   ",
    "/reset",
    "hi",
    "hi!",
    "thanks :)",
    "done???",
    "k8s",
    "yolo",
    "note this down",
    "hindsight",
    "why does the loader skip my provider",
]


@pytest.fixture(scope="module")
def upstream(repo_root: Path) -> Iterator[ModuleType]:
    """The real ``agent.memory_provider`` module, loaded straight from ``.refs/``.

    The upstream file is pure stdlib (``logging``, ``re``, ``abc``, ``dataclasses``,
    ``typing``), so it imports standalone with no Hermes package around it.

    It must be registered in ``sys.modules`` *before* execution: it uses
    ``from __future__ import annotations``, so ``RecallStatus``'s field annotations
    are strings, and ``dataclasses`` resolves them by looking the defining module up
    in ``sys.modules``. Registered under a private name and removed afterwards, so
    nothing else can accidentally import the reference copy.
    """
    path = repo_root / UPSTREAM_RELATIVE
    if not path.is_file():
        pytest.skip(
            f"reference tree absent ({UPSTREAM_RELATIVE}); expected in CI, "
            "where .refs/ is gitignored and not checked out"
        )
    module_name = "_hermes_recall_upstream_memory_provider"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def fallback_base() -> type:
    """The fallback ``MemoryProvider``, or a skip when the real Hermes is installed."""
    if _hermes.HERMES_AVAILABLE:
        pytest.skip("Hermes runtime is installed, so the fallback branch is not loaded")
    return _hermes.MemoryProvider


def test_trivial_prompt_regex_matches_upstream_exactly(upstream: ModuleType) -> None:
    if _hermes.HERMES_AVAILABLE:
        pytest.skip("Hermes runtime is installed; TRIVIAL_PROMPT_RE *is* upstream's")

    assert _hermes.TRIVIAL_PROMPT_RE.pattern == upstream.TRIVIAL_PROMPT_RE.pattern
    assert _hermes.TRIVIAL_PROMPT_RE.flags == upstream.TRIVIAL_PROMPT_RE.flags


@pytest.mark.parametrize("text", TRIVIAL_TABLE)
def test_is_trivial_prompt_agrees_with_upstream(upstream: ModuleType, text: str) -> None:
    assert _hermes.is_trivial_prompt(text) == upstream.is_trivial_prompt(text), text


def test_trivial_table_covers_both_verdicts(upstream: ModuleType) -> None:
    """A table that was all-True or all-False would prove nothing."""
    verdicts = {upstream.is_trivial_prompt(text) for text in TRIVIAL_TABLE}
    assert verdicts == {True, False}


def test_recall_status_dataclass_shape_matches_upstream(upstream: ModuleType) -> None:
    if _hermes.HERMES_AVAILABLE:
        pytest.skip("Hermes runtime is installed; RecallStatus *is* upstream's")

    ours = dataclasses.fields(_hermes.RecallStatus)
    theirs = dataclasses.fields(upstream.RecallStatus)

    assert [f.name for f in ours] == [f.name for f in theirs]
    assert [f.name for f in ours] == ["provider_label", "count", "glyph"]
    glyph_default = {f.name: f.default for f in ours}["glyph"]
    assert glyph_default == {f.name: f.default for f in theirs}["glyph"]
    assert _hermes.RecallStatus(provider_label="Recall", count=2).glyph == glyph_default


def test_indicator_glyph_matches_upstream(upstream: ModuleType) -> None:
    assert _hermes.INDICATOR_GLYPH == upstream.INDICATOR_GLYPH


def test_fallback_base_carries_every_hook_the_provider_overrides(
    upstream: ModuleType, fallback_base: type
) -> None:
    """The fallback must not drift *behind* the hooks ``RecallProvider`` implements."""
    overridden = {
        name
        for name in vars(RecallProvider)
        if not name.startswith("_") and hasattr(upstream.MemoryProvider, name)
    }
    # Sanity: if this set were empty the assertion below would be vacuous.
    assert "prefetch" in overridden
    assert "recall_status" in overridden

    missing = sorted(name for name in overridden if not hasattr(fallback_base, name))
    assert not missing, f"fallback MemoryProvider is missing hooks: {missing}"


def test_overridden_hook_signatures_match_upstream(upstream: ModuleType) -> None:
    """Same parameter names, kinds and defaults — a renamed keyword breaks dispatch."""
    for name in sorted(vars(RecallProvider)):
        if name.startswith("_"):
            continue
        theirs = getattr(upstream.MemoryProvider, name, None)
        if theirs is None:
            continue
        ours = getattr(RecallProvider, name)
        if isinstance(ours, property) or isinstance(theirs, property):
            continue
        assert _signature_shape(ours) == _signature_shape(theirs), name


def test_pre_compress_checkpoint_api_version_is_still_one(upstream: ModuleType) -> None:
    """Version 2 is the fail-closed checkpoint contract; this provider does not opt in."""
    assert upstream.MemoryProvider.pre_compress_checkpoint_api_version == 1
    assert RecallProvider.pre_compress_checkpoint_api_version == 1


def _signature_shape(func: object) -> list[tuple[str, str, object]]:
    """Parameter name, kind and default for each parameter, ignoring annotations."""
    signature = inspect.signature(func)  # type: ignore[arg-type]
    return [(param.name, str(param.kind), param.default) for param in signature.parameters.values()]
