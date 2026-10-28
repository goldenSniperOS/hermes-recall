"""Proxy for Engram's official MCP tools: one ``engram mcp --tools=agent`` stdio child.

recall does not hand-write the ``mem_*`` tools any more. It asks the ``engram``
binary which tools its agent profile serves, publishes those schemas to Hermes as
they are, and forwards every call to a long-lived MCP child it owns (S9, S10). The
v0.4 HTTP tools in ``tools.py`` stay as the fallback (S14).

The wire is MCP's stdio transport: newline-delimited JSON-RPC 2.0 — ``initialize``,
``notifications/initialized``, ``tools/list``, then ``tools/call {name, arguments}``.
A result is ``{content: [{type: "text", text}], isError?, structuredContent?}``.

**When Hermes asks.** ``MemoryManager.add_provider`` calls ``get_tool_schemas()``
before ``initialize`` and builds its tool routing table from that one answer, so
the tool list must exist synchronously on the first call, before recall knows its
session, profile or agent context. Discovery therefore runs a short-lived child
(spawn, list, stop) on the first call and caches the list twice: in-process, keyed
by the binary, and in ``<hermes_home>/recall/mcp-tools.json`` so a later process
with the same binary starts no child at all to answer the host.

**Concurrency.** The provider's worker and Hermes' tool thread run at the same
time, and one stdio child can serve one request at a time, so every call holds a
lock. A call is bounded by :data:`CALL_TIMEOUT_S`: a watchdog kills a child that
does not answer, which also unblocks a write stuck on a full pipe, so the lock can
never be held past that bound. A dead or killed child is replaced at most once per
proxy (per provider instance); a call that died after it was sent is never retried,
because Engram may already have applied it.

Stdlib only. Nothing here raises to the provider: every failure is a value.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .server import STATE_DIRNAME, _rotate_log, engram_bin

logger = logging.getLogger(__name__)

MCP_ARGS = ("mcp", "--tools=agent")
PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "hermes-recall"

# Bounds, read through the module at call time so tests can shrink them. A real
# Engram 3.6 child answers the handshake in well under a second and a local SQLite
# tool call in milliseconds; the call bound is generous because mem_doctor and a
# consolidation scan the store.
START_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 20.0
STOP_TIMEOUT_S = 1.0
_MAX_LIST_PAGES = 20

CACHE_FILENAME = "mcp-tools.json"
LOG_FILENAME = "engram-mcp.log"


class McpError(Exception):
    """The child failed to answer: exited, timed out, or sent nothing usable.

    ``delivered`` says whether the request had been written in full, i.e. whether
    Engram may have acted on it. ``timed_out`` tells a timeout from an exit.
    """

    def __init__(self, message: str, *, delivered: bool = False, timed_out: bool = False):
        super().__init__(message)
        self.delivered = delivered
        self.timed_out = timed_out


class McpRefusal(Exception):
    """A JSON-RPC ``error`` answer: the child is healthy, the request was refused."""


# -- one stdio child ------------------------------------------------------------------


class McpChild:
    """One ``engram mcp`` process and the JSON-RPC conversation with it."""

    def __init__(self, process: Any) -> None:
        self._process = process
        self._messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._next_id = 0
        self._eof = threading.Event()
        self._killed_by_watchdog = False
        reader = threading.Thread(target=self._read, name="recall-mcp-reader", daemon=True)
        reader.start()

    @classmethod
    def spawn(cls, binary: str, *, cwd: str = "", log_path: str = "") -> McpChild:
        """Launch ``<binary> mcp --tools=agent``. Raises :class:`McpError` when it cannot.

        Never a shell. The environment is inherited untouched (``ENGRAM_DATA_DIR``
        and friends must reach it, or it would open another store). stderr goes to
        ``log_path`` when it can be opened, else nowhere. Its own session on POSIX,
        so a Ctrl-C aimed at the agent's turn does not kill the memory child.
        """
        log_handle: Any = None
        if log_path:
            try:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                _rotate_log(Path(log_path))
                log_handle = open(log_path, "ab")  # noqa: SIM115 - closed below
            except OSError:
                log_handle = None
        options: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": log_handle if log_handle is not None else subprocess.DEVNULL,
            "close_fds": True,
            "cwd": cwd if cwd and os.path.isdir(cwd) else None,
        }
        if sys.platform != "win32":
            options["start_new_session"] = True
        try:
            process = subprocess.Popen([binary, *MCP_ARGS], **options)
        except OSError as exc:
            raise McpError(f'"{binary} mcp" could not be started ({exc.strerror or exc})') from None
        finally:
            if log_handle is not None:
                log_handle.close()
        return cls(process)

    @property
    def pid(self) -> int:
        return int(self._process.pid)

    def alive(self) -> bool:
        """Whether the process still runs and its stdout is still open."""
        try:
            return self._process.poll() is None and not self._eof.is_set()
        except Exception:
            return False

    def _read(self) -> None:
        """Reader thread: queue every JSON object the child prints, then ``None`` at EOF."""
        try:
            for raw in self._process.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    self._messages.put(message)
        except Exception:
            logger.debug("recall: MCP reader stopped", exc_info=True)
        finally:
            self._eof.set()
            self._messages.put(None)

    def _write(self, message: dict[str, Any]) -> None:
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        self._process.stdin.write(data)
        self._process.stdin.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        try:
            self._write(message)
        except (OSError, ValueError):
            raise McpError("Engram MCP exited") from None

    def request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        """Send one request and wait for its answer, killing the child at ``timeout``.

        The watchdog runs from before the write, so a child that stops reading its
        stdin cannot block the caller past the bound either.
        """
        self._next_id += 1
        request_id = self._next_id
        deadline = time.monotonic() + timeout
        watchdog = threading.Timer(max(0.0, timeout), self._watchdog)
        watchdog.daemon = True
        watchdog.start()
        delivered = False
        try:
            try:
                self._write(
                    {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
                )
                delivered = True
            except (OSError, ValueError):
                raise self._failure(delivered=False) from None
            while True:
                remaining = deadline - time.monotonic()
                try:
                    message = self._messages.get(timeout=max(0.0, remaining))
                except queue.Empty:
                    self.kill()
                    raise McpError("timed out", delivered=True, timed_out=True) from None
                if message is None:
                    raise self._failure(delivered=delivered)
                if message.get("id") != request_id or "method" in message:
                    continue  # a notification, or a server-initiated request
                error = message.get("error")
                if error is not None:
                    text = error.get("message") if isinstance(error, dict) else error
                    raise McpRefusal(str(text or "request refused"))
                return message.get("result")
        finally:
            watchdog.cancel()

    def _failure(self, *, delivered: bool) -> McpError:
        if self._killed_by_watchdog:
            return McpError("timed out", delivered=delivered, timed_out=True)
        return McpError("exited", delivered=delivered)

    def _watchdog(self) -> None:
        self._killed_by_watchdog = True
        self.kill()

    def initialize(self, timeout: float) -> dict[str, Any]:
        """The MCP handshake; returns the server's ``serverInfo`` (may be empty)."""
        result = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": __version__},
            },
            timeout,
        )
        self.notify("notifications/initialized")
        info = result.get("serverInfo") if isinstance(result, dict) else None
        return info if isinstance(info, dict) else {}

    def list_tools(self, timeout: float) -> list[dict[str, Any]]:
        """Every tool ``tools/list`` returns, following ``nextCursor`` pages."""
        deadline = time.monotonic() + timeout
        tools: list[dict[str, Any]] = []
        cursor: Any = None
        for _ in range(_MAX_LIST_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = self.request("tools/list", params, max(0.0, deadline - time.monotonic()))
            page = result.get("tools") if isinstance(result, dict) else None
            tools.extend(tool for tool in page or [] if isinstance(tool, dict))
            cursor = result.get("nextCursor") if isinstance(result, dict) else None
            if not cursor:
                break
        return tools

    def kill(self) -> None:
        try:
            if self._process.poll() is None:
                self._process.kill()
        except Exception:
            logger.debug("recall: could not kill the MCP child", exc_info=True)

    def stop(self) -> None:
        """Close stdin (Engram exits on EOF), wait briefly, then kill. Never raises."""
        try:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.wait(STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.kill()
                self._process.wait(STOP_TIMEOUT_S)
        except Exception:
            logger.debug("recall: MCP child stop failed", exc_info=True)


# -- discovery --------------------------------------------------------------------------


@dataclass(frozen=True)
class Discovery:
    """What ``tools/list`` returned for one binary, or why it could not be asked."""

    tools: tuple[dict[str, Any], ...] = ()
    version: str = ""
    error: str = ""


_DISCOVERY_CACHE: dict[tuple[str, int, int], Discovery] = {}
_DISCOVERY_LOCK = threading.Lock()


def clear_discovery_cache() -> None:
    """Forget every in-process discovery (a new process, as far as the cache knows)."""
    with _DISCOVERY_LOCK:
        _DISCOVERY_CACHE.clear()


def resolve_binary() -> str:
    """The ``engram`` the auto-start would run (``ENGRAM_BIN`` or ``PATH``), or ``""``."""
    try:
        return shutil.which(engram_bin()) or ""
    except Exception:
        return ""


def _fingerprint(binary: str) -> tuple[str, int, int] | None:
    """The binary's identity: resolved path, size and mtime.

    Stands in for "the Engram version" without running a process on every start:
    an upgrade (brew links a new Cellar path) or a rebuild changes at least one.
    """
    try:
        real = os.path.realpath(binary)
        info = os.stat(real)
    except OSError:
        return None
    return (real, int(info.st_size), int(info.st_mtime_ns))


def recall_dir(hermes_home: str) -> Path:
    return Path(hermes_home) / STATE_DIRNAME


def log_path(hermes_home: str) -> str:
    return str(recall_dir(hermes_home) / LOG_FILENAME) if hermes_home else ""


def discover(binary: str, hermes_home: str) -> Discovery:
    """The agent-profile tools of ``binary``, from cache or a short-lived child. Never raises.

    A failure is cached in-process too (a hung binary must not cost every provider
    instance another timeout), but never on disk.
    """
    try:
        fingerprint = _fingerprint(binary)
        if fingerprint is None:
            return Discovery(error=f'Engram binary "{binary}" not found')
        with _DISCOVERY_LOCK:
            cached = _DISCOVERY_CACHE.get(fingerprint)
            if cached is not None:
                return cached
            found = _read_cache(hermes_home, fingerprint)
            if found is None:
                found = _discover_live(binary, hermes_home)
                if not found.error:
                    _write_cache(hermes_home, fingerprint, found)
            _DISCOVERY_CACHE[fingerprint] = found
            return found
    except Exception as exc:  # noqa: BLE001 - the contract is "never raise"
        logger.debug("recall: MCP discovery failed", exc_info=True)
        return Discovery(error=f"MCP discovery failed: {type(exc).__name__}")


def _discover_live(binary: str, hermes_home: str) -> Discovery:
    deadline = time.monotonic() + START_TIMEOUT_S
    try:
        child = McpChild.spawn(binary, log_path=log_path(hermes_home))
    except McpError as exc:
        return Discovery(error=str(exc))
    try:
        info = child.initialize(START_TIMEOUT_S)
        tools = child.list_tools(max(0.0, deadline - time.monotonic()))
    except McpError as exc:
        reason = "timed out" if exc.timed_out else "exited"
        return Discovery(error=f'"{binary} mcp" {reason} before listing its tools')
    except McpRefusal as exc:
        return Discovery(error=f'"{binary} mcp" refused to list its tools: {exc}')
    finally:
        child.stop()
    valid = tuple(tool for tool in tools if isinstance(tool.get("name"), str) and tool["name"])
    if not valid:
        return Discovery(error=f'"{binary} mcp" listed no tools')
    version = info.get("version")
    return Discovery(tools=valid, version=version if isinstance(version, str) else "")


def _read_cache(hermes_home: str, fingerprint: tuple[str, int, int]) -> Discovery | None:
    if not hermes_home:
        return None
    try:
        payload = json.loads((recall_dir(hermes_home) / CACHE_FILENAME).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if [payload.get("binary"), payload.get("size"), payload.get("mtime_ns")] != list(fingerprint):
        return None
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    valid = tuple(t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str))
    version = payload.get("version")
    return Discovery(tools=valid, version=version if isinstance(version, str) else "")


def _write_cache(hermes_home: str, fingerprint: tuple[str, int, int], found: Discovery) -> None:
    """Atomically record the tool list for this binary. Never raises."""
    if not hermes_home:
        return
    try:
        directory = recall_dir(hermes_home)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "binary": fingerprint[0],
            "size": fingerprint[1],
            "mtime_ns": fingerprint[2],
            "version": found.version,
            "tools": list(found.tools),
            "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        fd, tmp_name = tempfile.mkstemp(prefix=".mcp-tools-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(tmp_name, directory / CACHE_FILENAME)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except Exception:
        logger.debug("recall: could not write the MCP tool cache (non-fatal)", exc_info=True)


def to_hermes_schema(tool: dict[str, Any]) -> dict[str, Any] | None:
    """An upstream MCP tool as Hermes' ``{name, description, parameters}``, or ``None``."""
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        return None
    description = tool.get("description")
    parameters = tool.get("inputSchema")
    return {
        "name": name,
        "description": description if isinstance(description, str) else "",
        "parameters": copy.deepcopy(parameters)
        if isinstance(parameters, dict)
        else {"type": "object", "properties": {}},
    }
