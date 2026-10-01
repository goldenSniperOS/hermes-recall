"""Run Hermes' real plugin install scanner over the files a git install would clone.

``hermes plugins install owner/repo`` scans the cloned tree with
``tools.plugin_guard.scan_plugin`` and treats every GitHub source as ``community``
trust. For community sources a ``caution`` verdict (any high-severity finding, for
example a privilege-escalation command named in a doc) blocks the install unless the user passes
``--force``. v0.1.0 shipped with exactly that problem, so this test keeps the tracked
tree at a ``safe`` verdict.

Like the loader test it runs in a subprocess with the Hermes checkout's interpreter,
so Hermes modules never leak into this process, and it skips cleanly when no Hermes
checkout or git checkout is available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_AGENT_DIR = Path.home() / ".hermes" / "hermes-agent"
HERMES_VENV_PYTHON = HERMES_AGENT_DIR / "venv" / "bin" / "python"
_SKIP_MARKER = "RECALL-SCAN-SKIP:"
_RESULT_MARKER = "RECALL-SCAN-RESULT:"

_CHILD = f"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
try:
    from tools.plugin_guard import scan_plugin
except Exception as exc:
    print({_SKIP_MARKER!r} + f"{{type(exc).__name__}}: {{exc}}")
    sys.exit(0)

result = scan_plugin(Path(sys.argv[2]), source="goldenSniperOS/hermes-recall")
print({_RESULT_MARKER!r} + json.dumps({{
    "verdict": result.verdict,
    "blocking": [
        f"{{f.severity}} {{f.pattern_id}} {{f.file}}:{{f.line}}"
        for f in result.findings
        if f.severity in ("high", "critical")
    ],
}}))
"""


def _tracked_files() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [name for name in completed.stdout.decode().split("\0") if name]


def _interpreters() -> list[str]:
    candidates = [sys.executable]
    if HERMES_VENV_PYTHON.is_file():
        candidates.append(str(HERMES_VENV_PYTHON))
    return candidates


def test_tracked_tree_passes_the_hermes_install_scan(tmp_path: Path) -> None:
    if not (HERMES_AGENT_DIR / "tools" / "plugin_guard.py").is_file():
        pytest.skip(f"no Hermes checkout with tools/plugin_guard.py at {HERMES_AGENT_DIR}")
    tracked = _tracked_files()
    if not tracked:
        pytest.skip("not a git checkout; cannot reproduce the cloned tree")

    clone = tmp_path / "recall"
    for name in tracked:
        source = REPO_ROOT / name
        if not source.is_file():
            continue
        target = clone / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    skip_reasons = []
    for python in _interpreters():
        completed = subprocess.run(
            [python, "-c", _CHILD, str(HERMES_AGENT_DIR), str(clone)],
            capture_output=True,
            text=True,
            timeout=120,
            env={"PYTHONDONTWRITEBYTECODE": "1", "HOME": str(tmp_path)},
        )
        for line in completed.stdout.splitlines():
            if line.startswith(_RESULT_MARKER):
                payload = json.loads(line[len(_RESULT_MARKER) :])
                assert payload["verdict"] == "safe", payload["blocking"]
                return
            if line.startswith(_SKIP_MARKER):
                skip_reasons.append(f"{python}: {line[len(_SKIP_MARKER) :]}")
                break
        else:
            pytest.fail(f"scanner child failed under {python}:\n{completed.stderr}")
    pytest.skip("; ".join(skip_reasons) or "no interpreter could import Hermes")
