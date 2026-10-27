"""Packaging invariants: version surfaces, manifest fields, entry point and discovery."""

from __future__ import annotations

import re
import tomllib

from conftest import REPO_ROOT

import hermes_recall

PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
MANIFEST = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
ROOT_INIT = (REPO_ROOT / "__init__.py").read_text(encoding="utf-8")


def manifest_field(key: str) -> str:
    """Read a scalar top-level key from plugin.yaml.

    Parsed with a regex on purpose: PyYAML is not a declared dependency of this
    plugin (runtime dependencies are empty), so the test suite must not import it.
    """
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(\S+)\s*$", MANIFEST)
    assert match, f"plugin.yaml has no {key}"
    return match.group(1).strip("\"'")


def test_all_version_surfaces_match() -> None:
    versions = {
        "pyproject.toml": PYPROJECT["project"]["version"],
        "plugin.yaml": manifest_field("version"),
        "hermes_recall.__version__": hermes_recall.__version__,
    }
    assert len(set(versions.values())) == 1, versions
    assert set(versions.values()) == {_latest_changelog_release()}, versions


def test_version_is_semver() -> None:
    assert re.fullmatch(r"^\d+\.\d+\.\d+$", hermes_recall.__version__)


def test_manifest_version_is_installable() -> None:
    # `hermes plugins install` (plugins_cmd._SUPPORTED_MANIFEST_VERSION) accepts
    # only manifest_version <= 1, although the loader reads v2 fields
    # (config_schema, license, ...) regardless of the number.
    raw = manifest_field("manifest_version")
    assert raw == "1"
    assert int(raw) <= 1


def test_manifest_kind_is_exclusive() -> None:
    # Only one memory provider can be active at a time (memory.provider in
    # config.yaml), so this plugin is exclusive rather than standalone.
    assert manifest_field("kind") == "exclusive"


def test_manifest_name_is_recall() -> None:
    # The memory loader never reads this field — it derives the provider name from
    # the install directory ($HERMES_HOME/plugins/<name>/) and reads only
    # `description` out of plugin.yaml. It still has to say `recall`, because the
    # install directory name, the `memory.provider` config value and
    # `RecallProvider.name` must all agree, and this manifest field is the
    # human-readable record of that single name.
    assert manifest_field("name") == "recall"


def test_pyproject_name_dependencies_and_entry_point() -> None:
    project = PYPROJECT["project"]
    assert project["name"] == "hermes-recall"
    assert project["dependencies"] == []
    entry_points = project["entry-points"]["hermes_agent.memory_providers"]
    assert entry_points == {"recall": "hermes_recall:register"}


def _latest_changelog_release() -> str:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## \[(\d+\.\d+\.\d+)\]", changelog, flags=re.M)
    assert match, "CHANGELOG.md has no released version section"
    return match.group(1)


def test_changelog_has_unreleased_and_first_release() -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in changelog
    assert "## [0.1.0]" in changelog


def test_root_init_matches_loader_discovery_heuristic() -> None:
    # `_is_memory_provider_dir` in plugins/memory/__init__.py text-scans the first
    # 8192 bytes of the plugin's __init__.py for `register_memory_provider` or
    # `MemoryProvider`. Fail that scan and the directory is skipped entirely —
    # the provider never shows up to be activated.
    assert "register_memory_provider" in ROOT_INIT[:8192]


def test_register_is_callable_and_reexported_by_root_init() -> None:
    assert callable(hermes_recall.register)
    # The root __init__.py is asserted on its source rather than imported: the
    # repository root is not itself a package, so there is no flat import that
    # reaches it, and the loader's synthetic `_hermes_user_memory.recall` name is
    # not reproducible here without re-implementing the loader. The invariant
    # under test is the single implementation: the root module imports `register`
    # from the `hermes_recall` package (both routes) and exports it.
    assert "from .hermes_recall import __version__, register" in ROOT_INIT
    assert "from hermes_recall import __version__, register" in ROOT_INIT
    assert '__all__ = ["__version__", "register"]' in ROOT_INIT


def test_manifest_provides_tools_matches_the_tool_schemas() -> None:
    from hermes_recall.tools import get_tool_schemas

    lines = MANIFEST.splitlines()
    assert "provides_tools:" in lines, "plugin.yaml has no block-style provides_tools list"
    listed: list[str] = []
    for line in lines[lines.index("provides_tools:") + 1 :]:
        item = re.fullmatch(r"\s+-\s*(\S+)\s*", line)
        if not item:
            break
        listed.append(item.group(1))
    assert listed == [schema["name"] for schema in get_tool_schemas()]
    assert len(listed) == 12


def test_root_cli_exposes_the_loader_hooks() -> None:
    # discover_plugin_cli_commands reads <provider dir>/cli.py, calls
    # register_cli(subparser) and resolves the handler as `<name>_command`.
    source = (REPO_ROOT / "cli.py").read_text(encoding="utf-8")
    assert "def register_cli(subparser" in source
    assert "def recall_command(args" in source
