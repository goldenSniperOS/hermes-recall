"""Engram server lifecycle: adopt a verified local ``engram serve``, or start one.

Ported from gentle-engram's ``initializeEngramServer`` (``index.ts``) so Hermes and
Pi treat the shared server the same way:

* **Externally managed** when ``ENGRAM_URL`` is set — never spawn (gentle-engram
  returns early on ``CONFIGURED_ENGRAM_URL``).
* **Only the address a plain ``engram serve`` binds** is ever spawned for:
  ``http://127.0.0.1:<port>`` (or ``localhost``), where ``<port>`` is
  ``ENGRAM_PORT`` or 7437. The Engram server listens on
  ``fmt.Sprintf("127.0.0.1:%d", s.port)`` (``server.go``), and the binary documents
  ``ENGRAM_PORT  Override HTTP server port (default: 7437)``. Any other URL could
  never be answered by the child we would launch.
* **Identity before trust.** A healthy server is adopted only when its
  ``/health`` ``instance_id`` equals ``ENGRAM_BIN instance-id``. A different id is a
  *foreign* instance; a server with no id fails closed exactly like gentle-engram
  (legacy or identity-missing): neither is adopted, spawned over or killed.
* **Detached spawn, never killed.** ``engram serve`` is started in its own session
  with no pipes back to us and is never waited on or terminated: it is shared with
  Pi and every other agent on the machine.
* **One start at a time** across processes: an exclusive non-blocking ``flock`` on
  ``<hermes_home>/recall/serve.lock``. A process that cannot take it only polls.

Every public function is fail-soft: nothing raises, every failure becomes an
``unavailable`` status with a human reason.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .client import EngramClient
from .config import DEFAULT_PORT, RecallConfig, engram_port, resolve_hermes_home

try:  # POSIX only; Windows has no fcntl and simply runs without the lock.
    import fcntl
except ImportError:  # pragma: no cover - exercised by monkeypatching to None
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "ADOPTED",
    "DISABLED",
    "EXTERNAL",
    "FOREIGN",
    "STARTED",
    "UNAVAILABLE",
    "ServerStatus",
    "engram_bin",
    "ensure_server",
    "expected_serve_port",
    "is_spawnable_url",
    "local_instance_id",
    "read_server_state",
    "write_server_state",
]

ADOPTED = "adopted"
STARTED = "started"
FOREIGN = "foreign"
UNAVAILABLE = "unavailable"
DISABLED = "disabled"
EXTERNAL = "external"
_STATUSES = frozenset({ADOPTED, STARTED, FOREIGN, UNAVAILABLE, DISABLED, EXTERNAL})

# Statuses that mean "recall vouches for this server", the only ones a later
# connection refusal may answer with the one bounded restart.
MANAGED_STATUSES = frozenset({ADOPTED, STARTED})

DEFAULT_SERVE_PORT = DEFAULT_PORT
_SPAWNABLE_HOSTS = frozenset({"127.0.0.1", "localhost"})

# Bounds. Read through the module at call time so tests can shrink them.
STARTUP_DEADLINE_S = 5.0
POLL_INTERVAL_S = 0.1
INSTANCE_ID_TIMEOUT_S = 3.0
HEALTH_PROBE_TIMEOUT_S = 0.5

STATE_DIRNAME = "recall"
LOCK_FILENAME = "serve.lock"
LOG_FILENAME = "engram-serve.log"
# The serve log is appended across starts; past this size it is rotated to
# ``engram-serve.log.1`` (replacing an older one) before the next start appends.
LOG_MAX_BYTES = 1024 * 1024
STATE_FILENAME = "server.json"

_INSTANCE_ID = re.compile(r"^[a-f0-9]{32}$")
_INSTALL_HINT = "install Engram or set ENGRAM_BIN"


@dataclass(frozen=True)
class ServerStatus:
    """How this provider obtained (or failed to obtain) its Engram server."""

    status: str
    reason: str = ""
    url: str = ""

    @property
    def managed(self) -> bool:
        """Whether recall vouches for this server (adopted or started it)."""
        return self.status in MANAGED_STATUSES


# -- address and binary ----------------------------------------------------------


def expected_serve_port(environ: Mapping[str, str] | None = None) -> int:
    """The port a plain ``engram serve`` binds: ``ENGRAM_PORT`` when valid, else 7437."""
    return engram_port(environ)


def is_spawnable_url(url: str, environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``url`` is exactly what a spawned ``engram serve`` would answer on.

    ``http`` only, host ``127.0.0.1`` or ``localhost`` (the server binds IPv4
    loopback, so ``[::1]`` would never reach it), an explicit port equal to
    :func:`expected_serve_port`, and no path (``/`` aside), query or fragment —
    ``engram serve`` routes from the root, so a prefixed URL is some other server.
    """
    try:
        parts = urllib.parse.urlsplit(str(url).strip())
        if parts.scheme != "http":
            return False
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            return False
        host = (parts.hostname or "").lower()
        if host not in _SPAWNABLE_HOSTS:
            return False
        return parts.port is not None and parts.port == expected_serve_port(environ)
    except ValueError:
        return False


def engram_bin(environ: Mapping[str, str] | None = None) -> str:
    """The Engram binary: ``ENGRAM_BIN`` when set, else ``engram`` on ``PATH``."""
    env = os.environ if environ is None else environ
    configured = str(env.get("ENGRAM_BIN", "") or "").strip()
    return configured or "engram"


def local_instance_id(binary: str, timeout: float | None = None) -> tuple[str, str]:
    """Run ``<binary> instance-id`` and return ``(id, "")`` or ``("", reason)``.

    Only stdout is read: the real binary prints an "Update available" notice on
    stderr. The failure messages follow gentle-engram's ``instanceIDFailureMessage``.
    """
    limit = INSTANCE_ID_TIMEOUT_S if timeout is None else timeout
    try:
        result = subprocess.run(
            [binary, "instance-id"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=max(0.1, limit),
            check=False,
        )
    except FileNotFoundError:
        return "", f'Engram binary "{binary}" not found; {_INSTALL_HINT}'
    except subprocess.TimeoutExpired:
        return "", (
            f'Engram binary "{binary}" did not answer "instance-id" within {limit:g}s; '
            "check for a hung or very slow binary"
        )
    except OSError as exc:
        return "", (
            f'Engram binary "{binary}" could not be started ({exc.strerror or exc}); '
            f"{_INSTALL_HINT}"
        )
    identifier = (result.stdout or "").strip() if result.returncode == 0 else ""
    if _INSTANCE_ID.match(identifier):
        return identifier, ""
    stderr = result.stderr or ""
    if result.returncode != 0 and re.search(r"unknown command", stderr, re.I):
        return "", (
            f'Engram binary "{binary}" does not support "instance-id" and predates '
            "v2.0.0-rc.11; upgrade it or point ENGRAM_BIN at the current one"
        )
    return "", (
        f'Engram binary "{binary}" failed to resolve its instance id '
        f'(exit {result.returncode}); run "{binary} instance-id" to see why'
    )


# -- health and identity ---------------------------------------------------------


def _probe(client: EngramClient) -> dict[str, Any] | None:
    """One short ``/health`` probe; the payload when it says ``ok``, else ``None``."""
    payload = client.health(timeout=HEALTH_PROBE_TIMEOUT_S)
    if isinstance(payload, dict) and payload.get("status") == "ok":
        return payload
    return None


def _is_pre_identity_version(version: str) -> bool:
    """Port of gentle-engram's ``isPreIdentityEngramVersion``: strictly < 2.0.0-rc.11."""
    match = re.fullmatch(
        r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
        r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?",
        version,
    )
    if not match:
        return False
    major, minor, patch = (int(part) for part in match.group(1, 2, 3))
    if major < 2:
        return True
    if (major, minor, patch) != (2, 0, 0):
        return False
    prerelease = match.group(4)
    if not prerelease:
        return False
    parts = prerelease.split(".")
    if any(p.isdigit() and len(p) > 1 and p.startswith("0") for p in parts):
        return False
    identity_release = ["rc", "11"]
    for index in range(max(len(parts), len(identity_release))):
        if index >= len(parts):
            return True
        if index >= len(identity_release):
            return False
        actual, expected = parts[index], identity_release[index]
        if actual == expected:
            continue
        if actual.isdigit() and expected.isdigit():
            return int(actual) < int(expected)
        if actual.isdigit() != expected.isdigit():
            return actual.isdigit()
        return actual < expected
    return False


def _classify(health: dict[str, Any], expected_id: str, url: str, how: str) -> ServerStatus:
    """Turn a healthy ``/health`` body into adopted / started / foreign / unavailable.

    ``foreign`` is reserved for a definite identity **mismatch**, and the provider
    sends such a server no memory traffic. A server with no identity at all is
    ``unavailable`` (used, never managed): a deliberate divergence from
    gentle-engram, which fails closed there — see ``docs/design.md``.
    """
    served = health.get("instance_id")
    version = health.get("version")
    version_text = version.strip() if isinstance(version, str) and version.strip() else "unknown"
    if not isinstance(served, str) or not served:
        if _is_pre_identity_version(version_text):
            reason = (
                f"Engram server at {url} predates instance identity (server {version_text}); "
                "used but not managed. Stop it and start the current binary when you can. "
                "Nothing is terminated automatically."
            )
        else:
            reason = (
                f"Engram server at {url} did not report its instance identity "
                f"(server {version_text}); used but not managed. Verify or upgrade it. "
                "Nothing is terminated automatically."
            )
        return ServerStatus(UNAVAILABLE, reason, url)
    if served != expected_id:
        return ServerStatus(
            FOREIGN,
            f"Engram server at {url} is a different instance than ENGRAM_BIN "
            f"(instance {served}); memory is not sent to it, and nothing is terminated",
            url,
        )
    if how == STARTED:
        return ServerStatus(STARTED, f"spawned engram serve for {url}", url)
    return ServerStatus(ADOPTED, f"matching Engram instance already running at {url}", url)


def _wait_for_health(
    client: EngramClient,
    deadline: float,
    should_stop: Callable[[], bool],
    child: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    """Poll ``/health`` until it answers ok or the deadline passes.

    Returns ``(payload, "")`` on success, ``(None, reason)`` otherwise. ``child`` is
    only *polled* (which reaps an early exit); it is never waited on or killed.
    """
    exited = ""
    while True:
        if should_stop():
            return None, "provider shut down while waiting for Engram"
        payload = _probe(client)
        if payload is not None:
            return payload, ""
        if child is not None and not exited:
            try:
                code = child.poll()
            except Exception:
                code = None
            if code is not None:
                exited = f"engram serve exited with code {code}"
        if time.monotonic() >= deadline:
            return None, exited
        time.sleep(POLL_INTERVAL_S)


def _exited(child: Any) -> bool:
    """Whether the spawned child has already exited (best effort, never raises)."""
    try:
        return child.poll() is not None
    except Exception:
        return False


# -- spawn and lock --------------------------------------------------------------


def _recall_dir(hermes_home: str) -> Path:
    return Path(resolve_hermes_home(hermes_home)) / STATE_DIRNAME


def _acquire_lock(hermes_home: str) -> tuple[Any, bool]:
    """Try the cross-process start lock. Returns ``(handle, acquired)``.

    ``(None, True)`` when locking is unavailable (no ``fcntl``, unwritable home):
    the lock is a courtesy between processes, and its absence must not stop the
    start. ``(None, False)`` only when another holder has it.
    """
    if fcntl is None:
        return None, True
    try:
        directory = _recall_dir(hermes_home)
        directory.mkdir(parents=True, exist_ok=True)
        handle = open(directory / LOCK_FILENAME, "a+")  # noqa: SIM115 - held by caller
    except OSError:
        logger.debug("recall: cannot open the serve lock; starting without it", exc_info=True)
        return None, True
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None, False
    return handle, True


def _release_lock(handle: Any) -> None:
    if handle is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        try:
            handle.close()
        except OSError:
            pass


def _popen(args: list[str], **options: Any) -> Any:
    """The one place a process is launched for ``serve`` (a seam for tests).

    Separate from ``subprocess.run`` on purpose: patching ``subprocess.Popen`` itself
    would also break the ``instance-id`` call, which runs through it.
    """
    return subprocess.Popen(args, **options)


def _rotate_log(log_path: Path) -> None:
    """Move an oversized serve log to ``.1`` so appends stay bounded. Never raises."""
    try:
        if log_path.stat().st_size > LOG_MAX_BYTES:
            os.replace(log_path, log_path.with_name(log_path.name + ".1"))
    except OSError:
        pass


def _reap(child: Any) -> None:
    """Wait for ``child`` on a daemon thread so its exit never leaves a zombie.

    This only collects the exit status; it never signals the process. Without it a
    long-lived gateway would keep a defunct ``engram serve`` entry forever after
    the server exits on its own.
    """

    def wait() -> None:
        try:
            child.wait()
        except Exception:
            logger.debug("recall: reaping engram serve failed", exc_info=True)

    threading.Thread(target=wait, name="recall-engram-reaper", daemon=True).start()


def _spawn_serve(binary: str, hermes_home: str) -> Any:
    """Launch ``<binary> serve`` fully detached. Raises ``OSError`` when it cannot.

    No pipes back to us, its own session (POSIX) or a detached process group
    (Windows), inherited fds closed, cwd the recall state directory (never the
    user's repository). The environment is inherited on purpose: ``ENGRAM_PORT``
    and Engram's data-dir settings must reach the child, or it would serve another
    port or store. It is a copy plus ``ENGRAM_CLOUD_AUTOSYNC=1``, as gentle-engram
    spawns it (2f1e545); the server skips autosync itself when no cloud server or
    token is configured, and our own ``os.environ`` is never touched. Output goes
    to ``<hermes_home>/recall/engram-serve.log`` when that can be opened. The child
    is never killed: it outlives this process on purpose; a daemon thread only reaps
    its exit status (:func:`_reap`).
    """
    log_handle: Any = None
    directory: Path | None = None
    try:
        directory = _recall_dir(hermes_home)
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        directory = None
    if directory is not None:
        log_path = directory / LOG_FILENAME
        _rotate_log(log_path)
        try:
            log_handle = open(log_path, "ab")  # noqa: SIM115 - closed below
        except OSError:
            log_handle = None
    output: Any = log_handle if log_handle is not None else subprocess.DEVNULL
    options: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": output,
        "stderr": subprocess.STDOUT if log_handle is not None else subprocess.DEVNULL,
        "close_fds": True,
        "cwd": str(directory) if directory is not None else None,
        "env": {**os.environ, "ENGRAM_CLOUD_AUTOSYNC": "1"},
    }
    if sys.platform == "win32":  # pragma: no cover - not exercised on POSIX CI
        options["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        options["start_new_session"] = True
    try:
        child = _popen([binary, "serve"], **options)
    finally:
        if log_handle is not None:
            log_handle.close()
    _reap(child)
    return child


# -- entry point -----------------------------------------------------------------


def ensure_server(
    config: RecallConfig,
    hermes_home: str = "",
    *,
    client: EngramClient | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ServerStatus:
    """Make sure a verified local Engram answers at ``config.url``. Never raises.

    See the module docstring and ``docs/design.md`` ("Engram server lifecycle") for
    the decision table. Bounded: at most :data:`INSTANCE_ID_TIMEOUT_S` for the
    identity plus :data:`STARTUP_DEADLINE_S` of readiness polling.
    """
    url = ""
    try:
        url = str(config.url).strip().rstrip("/")
        stop = should_stop or (lambda: False)

        if str(os.environ.get("ENGRAM_URL", "") or "").strip():
            return ServerStatus(
                EXTERNAL, "ENGRAM_URL is set; the server is managed externally", url
            )
        if not getattr(config, "autostart", True):
            return ServerStatus(DISABLED, "autostart is off (memory.recall.autostart)", url)
        if not is_spawnable_url(url):
            return ServerStatus(
                EXTERNAL,
                f"{url} is not the address `engram serve` binds "
                f"(http://127.0.0.1:{expected_serve_port()}); not managed",
                url,
            )

        probe_client = client if client is not None else EngramClient(config)
        binary = engram_bin()
        health = _probe(probe_client)
        expected_id, id_error = local_instance_id(binary)

        if health is not None:
            if not expected_id:
                return ServerStatus(
                    UNAVAILABLE, f"cannot verify the running server: {id_error}", url
                )
            return _classify(health, expected_id, url, ADOPTED)

        if not expected_id:
            return ServerStatus(UNAVAILABLE, id_error, url)

        deadline = time.monotonic() + STARTUP_DEADLINE_S
        handle, acquired = _acquire_lock(hermes_home)
        if not acquired:
            payload, _ = _wait_for_health(probe_client, deadline, stop)
            if payload is None:
                return ServerStatus(
                    UNAVAILABLE,
                    "another recall process holds the start lock and Engram did not "
                    f"answer within {STARTUP_DEADLINE_S:g}s",
                    url,
                )
            return _classify(payload, expected_id, url, ADOPTED)

        try:
            # Someone may have started it between our probe and the lock.
            again = _probe(probe_client)
            if again is not None:
                return _classify(again, expected_id, url, ADOPTED)
            if stop():
                return ServerStatus(
                    UNAVAILABLE, "provider shut down before Engram was started", url
                )
            try:
                child = _spawn_serve(binary, hermes_home)
            except OSError as exc:
                return ServerStatus(
                    UNAVAILABLE,
                    f'could not start "{binary} serve" ({exc.strerror or exc}); {_INSTALL_HINT}',
                    url,
                )
            payload, why = _wait_for_health(probe_client, deadline, stop, child=child)
        finally:
            _release_lock(handle)

        if payload is None:
            detail = f" ({why})" if why else ""
            return ServerStatus(
                UNAVAILABLE,
                f"engram serve did not answer within {STARTUP_DEADLINE_S:g}s{detail}",
                url,
            )
        # A matching server answers, but our child already exited on its own: the
        # port was taken by an instance someone else started, so it is adopted.
        return _classify(payload, expected_id, url, ADOPTED if _exited(child) else STARTED)
    except Exception as exc:  # noqa: BLE001 - the contract is "never raise"
        logger.debug("recall: ensure_server failed (non-fatal)", exc_info=True)
        return ServerStatus(UNAVAILABLE, f"auto-start failed: {type(exc).__name__}", url)


# -- persisted status (read by `hermes recall status` in another process) --------


def server_state_path(hermes_home: str) -> Path:
    """``<hermes_home>/recall/server.json``."""
    return _recall_dir(hermes_home) / STATE_FILENAME


def write_server_state(hermes_home: str, status: ServerStatus) -> None:
    """Atomically record the last lifecycle outcome. Never raises."""
    try:
        target = server_state_path(hermes_home)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": status.status,
            "reason": status.reason,
            "url": status.url,
            "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        fd, tmp_name = tempfile.mkstemp(prefix=".server-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except Exception:
        logger.debug("recall: could not write server state (non-fatal)", exc_info=True)


def read_server_state(hermes_home: str) -> ServerStatus | None:
    """The recorded lifecycle outcome, or ``None`` when missing or corrupt. Never raises."""
    try:
        payload = json.loads(server_state_path(hermes_home).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") not in _STATUSES:
        return None
    reason = payload.get("reason")
    url = payload.get("url")
    return ServerStatus(
        str(payload["status"]),
        reason if isinstance(reason, str) else "",
        url if isinstance(url, str) else "",
    )
