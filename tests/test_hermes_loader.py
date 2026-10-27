"""Load the plugin through the real Hermes memory loader, when Hermes is installed.

The other compat tests compare against vendored excerpts in ``.refs/``. This one runs
the actual ``plugins.memory.load_memory_provider("recall")`` from a local Hermes
checkout against a copy of this repository installed as a user plugin, which is the
only way to prove the directory layout, the root ``__init__.py`` and the relative
imports survive the loader's synthetic-package import.

It runs in a **subprocess**: putting the Hermes checkout on this process's
``sys.path`` would make ``agent.*`` importable for every later test, silently
switching ``hermes_recall._hermes`` and the skill extractor to the real Hermes
modules. The child also gets its own ``HOME`` and ``HERMES_HOME`` under ``tmp_path``
and an ``ENGRAM_URL`` with nothing listening, so nothing touches the user's real
Hermes home or Engram.

The child interpreter is this one first (the brief's ``.venv``), then the Hermes
checkout's own ``venv`` when this one lacks Hermes' dependencies (it needs PyYAML,
which this stdlib-only project does not install). Skips cleanly when no checkout is
present at ``~/.hermes/hermes-agent`` or when no candidate interpreter can import it.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

HERMES_AGENT_DIR = Path.home() / ".hermes" / "hermes-agent"
HERMES_VENV_PYTHON = HERMES_AGENT_DIR / "venv" / "bin" / "python"
EXPECTED_TOOL_COUNT = 12
_SKIP_MARKER = "RECALL-LOADER-SKIP:"
_RESULT_MARKER = "RECALL-LOADER-RESULT:"

_IGNORED = shutil.ignore_patterns(
    ".git",
    ".venv",
    ".refs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".codegraph",
    "*.pyc",
)

_CHILD = f"""
import json
import sys

sys.path.insert(0, sys.argv[1])
try:
    from plugins.memory import load_memory_provider
except Exception as exc:  # a missing Hermes dependency is a skip, not a failure
    print({_SKIP_MARKER!r} + f"{{type(exc).__name__}}: {{exc}}")
    sys.exit(0)

provider = load_memory_provider("recall")
if provider is None:
    print({_RESULT_MARKER!r} + json.dumps({{"loaded": False}}))
    sys.exit(0)
schemas = provider.get_tool_schemas()
print({_RESULT_MARKER!r} + json.dumps({{
    "loaded": True,
    "name": provider.name,
    "tools": len(schemas),
    "module": type(provider).__module__,
}}))
"""


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(
    not (HERMES_AGENT_DIR / "plugins" / "memory" / "__init__.py").is_file(),
    reason=f"no Hermes checkout at {HERMES_AGENT_DIR}",
)
def test_the_real_hermes_loader_loads_recall(repo_root: Path, tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes-home"
    plugin_dir = hermes_home / "plugins" / "recall"
    shutil.copytree(repo_root, plugin_dir, ignore=_IGNORED)

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "ENGRAM_URL", "HERMES_HOME"}
    }
    env.update(
        {
            "HOME": str(tmp_path),
            "HERMES_HOME": str(hermes_home),
            "ENGRAM_URL": f"http://127.0.0.1:{_unused_port()}",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    interpreters = [sys.executable]
    if HERMES_VENV_PYTHON.is_file():
        interpreters.append(str(HERMES_VENV_PYTHON))

    skipped: list[str] = []
    for interpreter in interpreters:
        completed = subprocess.run(
            [interpreter, "-c", _CHILD, str(HERMES_AGENT_DIR)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        output = completed.stdout.splitlines()
        skip = [line for line in output if line.startswith(_SKIP_MARKER)]
        if not skip:
            break
        skipped.append(f"{interpreter}: {skip[0][len(_SKIP_MARKER) :]}")
    else:
        pytest.skip("Hermes is not importable here: " + "; ".join(skipped))

    results = [line for line in output if line.startswith(_RESULT_MARKER)]
    assert completed.returncode == 0, completed.stderr
    assert results, f"loader produced no result\nstdout={completed.stdout}\n{completed.stderr}"
    result = json.loads(results[-1][len(_RESULT_MARKER) :])

    assert result["loaded"] is True, completed.stderr
    assert result["name"] == "recall"
    assert result["tools"] == EXPECTED_TOOL_COUNT
    assert result["module"].endswith("hermes_recall.provider")
