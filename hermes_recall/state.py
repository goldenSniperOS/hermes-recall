"""Per-profile JSON state on disk: the last memory id this provider touched.

**Stable contract.** Other plugins read this file (hermes-odd shows the last memory
id in ``/odd_status``), so its location, keys and types are frozen:

* Path: ``<hermes_home>/recall/state.json``.
* Content: one JSON object with exactly these keys:

  ===================  =======  ================================================
  key                  type     meaning
  ===================  =======  ================================================
  ``last_memory_id``   int      Engram observation id of the last tool result
  ``last_action``      str      ``saved``, ``updated``, ``observation``,
                                ``pinned``, ``unpinned``, ``session_summary`` or
                                ``consolidated``
  ``project``          str      Engram project the action ran under (may be "")
  ``updated_at``       str      ISO 8601 UTC timestamp of the write
  ===================  =======  ================================================

* Writes are atomic: a temp file in the same directory, then ``os.replace``, so a
  reader never sees a half-written file.

Public API: :func:`read_state` and :func:`write_state`. Stdlib only, no Hermes
import, so another plugin may import this module on its own.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["STATE_DIRNAME", "STATE_FILENAME", "read_state", "state_path", "write_state"]

STATE_DIRNAME = "recall"
STATE_FILENAME = "state.json"


def state_path(hermes_home: str | os.PathLike[str]) -> Path:
    """Return ``<hermes_home>/recall/state.json``."""
    return Path(hermes_home) / STATE_DIRNAME / STATE_FILENAME


def read_state(hermes_home: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the persisted state, or ``{}`` when it is missing or unreadable.

    Never raises. A file that is not a JSON object, or whose values have the wrong
    types, counts as corrupt and yields ``{}`` — a reader must be able to trust
    every key it gets back.
    """
    try:
        raw = state_path(hermes_home).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    last_id = payload.get("last_memory_id")
    if isinstance(last_id, bool) or not isinstance(last_id, int):
        return {}
    for key in ("last_action", "project", "updated_at"):
        if not isinstance(payload.get(key), str):
            return {}
    return {
        "last_memory_id": last_id,
        "last_action": payload["last_action"],
        "project": payload["project"],
        "updated_at": payload["updated_at"],
    }


def write_state(
    hermes_home: str | os.PathLike[str],
    *,
    last_memory_id: int,
    last_action: str,
    project: str,
) -> None:
    """Atomically replace the state file. Raises ``OSError``/``ValueError`` on failure.

    The caller decides whether a failure matters; the provider swallows it.
    """
    if isinstance(last_memory_id, bool) or not isinstance(last_memory_id, int):
        raise ValueError("last_memory_id must be an int")
    payload = {
        "last_memory_id": last_memory_id,
        "last_action": str(last_action),
        "project": str(project or ""),
        "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    target = state_path(hermes_home)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
