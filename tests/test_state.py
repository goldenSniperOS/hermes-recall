"""The frozen ``state.json`` contract: path, exact keys and types, atomicity, read_state."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from hermes_recall import state as state_module
from hermes_recall.config import resolve_hermes_home
from hermes_recall.state import read_state, state_path, write_state


def test_state_path_is_hermes_home_recall_state_json(tmp_path: Path) -> None:
    assert state_path(tmp_path) == tmp_path / "recall" / "state.json"
    assert state_path(str(tmp_path)) == tmp_path / "recall" / "state.json"


def test_write_state_produces_exactly_the_contract_keys_and_types(tmp_path: Path) -> None:
    write_state(tmp_path, last_memory_id=761, last_action="saved", project="hermes-recall")

    payload = json.loads((tmp_path / "recall" / "state.json").read_text(encoding="utf-8"))

    assert set(payload) == {"last_memory_id", "last_action", "project", "updated_at"}
    assert payload["last_memory_id"] == 761
    assert type(payload["last_memory_id"]) is int
    assert payload["last_action"] == "saved"
    assert payload["project"] == "hermes-recall"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", payload["updated_at"])


def test_read_state_round_trips(tmp_path: Path) -> None:
    write_state(tmp_path, last_memory_id=1, last_action="pinned", project="")
    state = read_state(tmp_path)
    assert state["last_memory_id"] == 1
    assert state["last_action"] == "pinned"
    assert state["project"] == ""


def test_write_state_replaces_atomically_via_os_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src: Any, dst: Any) -> None:
        calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(state_module.os, "replace", spy)
    write_state(tmp_path, last_memory_id=2, last_action="saved", project="p")

    assert len(calls) == 1
    src, dst = calls[0]
    assert Path(src).parent == tmp_path / "recall"  # temp file in the same directory
    assert dst == str(tmp_path / "recall" / "state.json")
    assert sorted(p.name for p in (tmp_path / "recall").iterdir()) == ["state.json"]


def test_a_failed_replace_leaves_the_old_file_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_state(tmp_path, last_memory_id=3, last_action="saved", project="p")

    def boom(_src: Any, _dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(state_module.os, "replace", boom)
    with pytest.raises(OSError):
        write_state(tmp_path, last_memory_id=4, last_action="saved", project="p")

    assert read_state(tmp_path)["last_memory_id"] == 3
    assert sorted(p.name for p in (tmp_path / "recall").iterdir()) == ["state.json"]


def test_write_state_rejects_a_non_int_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_state(tmp_path, last_memory_id=True, last_action="saved", project="p")
    with pytest.raises(ValueError):
        write_state(tmp_path, last_memory_id="761", last_action="saved", project="p")  # type: ignore[arg-type]


def test_read_state_on_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_state(tmp_path) == {}
    assert read_state(tmp_path / "does-not-exist") == {}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "{not json",
        "[]",
        '"text"',
        '{"last_memory_id": "761", "last_action": "saved", "project": "", "updated_at": ""}',
        '{"last_memory_id": true, "last_action": "saved", "project": "", "updated_at": ""}',
        '{"last_memory_id": 761, "last_action": 1, "project": "", "updated_at": ""}',
        '{"last_memory_id": 761, "last_action": "saved"}',
    ],
)
def test_read_state_on_corrupt_file_is_empty(tmp_path: Path, raw: str) -> None:
    target = tmp_path / "recall" / "state.json"
    target.parent.mkdir(parents=True)
    target.write_text(raw, encoding="utf-8")
    assert read_state(tmp_path) == {}


def test_read_state_never_raises_on_a_bad_path() -> None:
    assert read_state("\0bad") == {}


def test_resolve_hermes_home_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "env"))
    assert resolve_hermes_home(str(tmp_path / "kw")) == str(tmp_path / "kw")
    assert resolve_hermes_home("") == str(tmp_path / "env")
    monkeypatch.delenv("HERMES_HOME")
    assert resolve_hermes_home() == os.path.join(os.path.expanduser("~"), ".hermes")
