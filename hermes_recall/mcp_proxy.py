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
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from . import tools as tools_module
from .config import RecallConfig
from .format import (
    consolidated_status,
    observation_status,
    pinned_status,
    saved_status,
    unpinned_status,
    updated_status,
)
from .redact import PRIVATE_TAG_RE, strip_private
from .server import STATE_DIRNAME, _rotate_log, engram_bin, is_spawnable_url

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
# The tool source this profile's provider last chose, for ``hermes recall status``
# (a separate process with no provider of its own).
SOURCE_FILENAME = "tools.json"

# Where the tool schemas came from (``hermes recall status`` names it).
TOOLS_MCP = "mcp"
TOOLS_HTTP = "http"

# S12: the only tools a subagent, cron or flush context may call. An explicit
# allowlist, not Engram's ``readOnlyHint``: any other tool — including one a future
# Engram adds — counts as a write and is refused before the child is touched.
READ_TOOLS = frozenset(
    {
        "mem_search",
        "mem_search_prompts",
        "mem_context",
        "mem_get_observation",
        "mem_current_project",
        "mem_list_projects",
        "mem_suggest_topic_key",
        "mem_doctor",
    }
)

# Tools whose ``session_id`` / ``id`` is recall's session: their success (or an
# uncertain outcome) ends it, so the hook's ``/end`` is not posted (S13).
SESSION_ENDING_TOOLS = frozenset({"mem_session_finalize", "mem_session_end"})


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


# -- the tool catalog the provider publishes ---------------------------------------------


@dataclass(frozen=True)
class ToolCatalog:
    """The tool set this provider publishes: Engram's MCP tools or the v0.4 HTTP ones."""

    source: str
    reason: str = ""
    schemas: tuple[dict[str, Any], ...] = ()
    binary: str = ""
    version: str = ""
    names: frozenset[str] = field(default=frozenset())

    def properties(self, name: str) -> frozenset[str]:
        """The parameter names tool ``name`` declares (what may be injected into it)."""
        for schema in self.schemas:
            if schema["name"] == name:
                props = schema.get("parameters", {}).get("properties")
                return frozenset(props) if isinstance(props, dict) else frozenset()
        return frozenset()

    def label(self) -> str:
        """One status line: which tools the model has, and why when it is the fallback."""
        count = f"{len(self.schemas)} tools"
        if self.source == TOOLS_MCP:
            return f"Engram MCP, {count} (Engram {self.version or 'unknown'})"
        return f"HTTP fallback, {count}: {self.reason}" if self.reason else f"HTTP, {count}"

    @classmethod
    def build(cls, source: str, schemas: list[dict[str, Any]], **fields: Any) -> ToolCatalog:
        return cls(
            source=source,
            schemas=tuple(schemas),
            names=frozenset(schema["name"] for schema in schemas),
            **fields,
        )


def _allowed(schemas: list[dict[str, Any]], allowlist: tuple[str, ...]) -> list[dict[str, Any]]:
    if not allowlist:
        return schemas
    return [schema for schema in schemas if schema["name"] in allowlist]


def http_catalog(config: RecallConfig, reason: str) -> ToolCatalog:
    """The v0.4 HTTP tool set (``tools.py``), unchanged apart from the allowlist."""
    schemas = _allowed(tools_module.get_tool_schemas(), config.tools)
    return ToolCatalog.build(TOOLS_HTTP, schemas, reason=reason)


def build_catalog(config: RecallConfig, hermes_home: str) -> ToolCatalog:
    """Decide the tool source once per provider. Never raises.

    The proxy needs all of (S14): no ``ENGRAM_URL`` and a configured URL that is the
    local default address — the MCP child opens the local SQLite store directly, so
    its writes must land where the HTTP server recall talks to reads — plus an
    ``engram`` binary that lists its tools. Otherwise the v0.4 HTTP tools, with the
    reason. (A foreign instance is only known after ``initialize``; the provider then
    refuses every tool call, whichever source was published.)
    """
    try:
        if str(os.environ.get("ENGRAM_URL", "") or "").strip():
            return http_catalog(
                config,
                "ENGRAM_URL is set; Engram's MCP opens the local store, so it is not "
                "paired with an externally managed server",
            )
        url = str(config.url).strip()
        if not is_spawnable_url(url):
            return http_catalog(
                config,
                f"{url} is not the local default Engram address; Engram's MCP opens the "
                "local store, so it is only paired with that server",
            )
        binary = resolve_binary()
        if not binary:
            return http_catalog(config, f'Engram binary "{engram_bin()}" not found')
        discovery = discover(binary, hermes_home)
        if discovery.error:
            return http_catalog(config, discovery.error)
        mapped = [to_hermes_schema(tool) for tool in discovery.tools]
        schemas = _allowed([schema for schema in mapped if schema], config.tools)
        return ToolCatalog.build(
            TOOLS_MCP,
            schemas,
            reason=f"engram mcp {MCP_ARGS[1]} (Engram {discovery.version or 'unknown'})",
            binary=binary,
            version=discovery.version,
        )
    except Exception as exc:  # noqa: BLE001 - the contract is "never raise"
        logger.debug("recall: tool catalog failed; using the HTTP tools", exc_info=True)
        return http_catalog(config, f"tool discovery failed: {type(exc).__name__}")


def write_tool_source(hermes_home: str, catalog: ToolCatalog) -> None:
    """Record the chosen tool source in ``<hermes_home>/recall/tools.json``. Never raises."""
    if not hermes_home:
        return
    try:
        directory = recall_dir(hermes_home)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": catalog.source,
            "reason": catalog.reason,
            "count": len(catalog.schemas),
            "version": catalog.version,
            "label": catalog.label(),
            "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        fd, tmp_name = tempfile.mkstemp(prefix=".tools-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(tmp_name, directory / SOURCE_FILENAME)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except Exception:
        logger.debug("recall: could not record the tool source (non-fatal)", exc_info=True)


def read_tool_source(hermes_home: str) -> dict[str, Any] | None:
    """The recorded tool source, or ``None`` when missing or unreadable. Never raises."""
    try:
        payload = json.loads((recall_dir(hermes_home) / SOURCE_FILENAME).read_text("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("source") not in (TOOLS_MCP, TOOLS_HTTP):
        return None
    return payload


def recorded_label(hermes_home: str) -> str:
    """The status line for the recorded tool source (``hermes recall status``)."""
    recorded = read_tool_source(hermes_home)
    label = recorded.get("label") if recorded is not None else None
    return label if isinstance(label, str) and label else "— (not recorded yet)"


# -- calls ------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallOutcome:
    """One proxied call: the JSON payload for the model plus what happened.

    ``ok``: Engram answered without ``isError``. ``refused``: Engram answered with
    ``isError`` or a JSON-RPC error. ``uncertain``: the request reached the child but
    no answer came back, so Engram may have applied it.
    """

    payload: dict[str, Any]
    ok: bool = False
    refused: bool = False
    uncertain: bool = False


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def render_result(result: Any) -> tuple[dict[str, Any], bool]:
    """An MCP ``tools/call`` result as the JSON object handed to the model.

    Structured content wins; otherwise the text items, joined, parsed as a JSON
    object when they are one (Engram's results are), else ``{"result": text}``. An
    ``isError`` result becomes ``{"error": <message>}`` plus the other fields of a
    JSON error body (``error_code``, ``hint``, ``available_projects``, ...). Returns
    ``(payload, is_error)``.
    """
    if not isinstance(result, dict):
        return {"result": result}, False
    texts = [
        item["text"]
        for item in result.get("content") or []
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    ]
    text = "\n".join(texts)
    if result.get("isError") is True:
        body = _json_object(text)
        if body is None:
            return {"error": text or "Engram reported an error"}, True
        message = body.get("message") or body.get("error") or text
        rest = {key: value for key, value in body.items() if key not in ("message", "error")}
        return {"error": message, **rest}, True
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured, False
    body = _json_object(text)
    return (body if body is not None else {"result": text}), False


def canonical_project(name: str) -> str:
    """Engram's ``CanonicalizeProjectName``: NFC, lower-case, trimmed, no doubled separators."""
    canonical = unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).lower().strip())
    while "--" in canonical:
        canonical = canonical.replace("--", "-")
    while "__" in canonical:
        canonical = canonical.replace("__", "_")
    return canonical


def redact_private(value: Any) -> Any:
    """``value`` with every ``<private>`` block in every string replaced, recursively.

    Only strings that carry a tag are touched (``strip_private`` also trims), so ids,
    keys and every other argument reach Engram byte for byte.
    """
    if isinstance(value, str):
        return strip_private(value) if PRIVATE_TAG_RE.search(value) else value
    if isinstance(value, dict):
        return {key: redact_private(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_private(item) for item in value]
    return value


def _observation_id(value: Any) -> int | None:
    if isinstance(value, str) and value.strip().lstrip("#").isdigit():
        value = int(value.strip().lstrip("#"))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


@dataclass(frozen=True)
class Effect:
    """The observation a successful call touched: its id, ``state.json`` action, status."""

    observation_id: int
    action: str
    status: str
    project: str = ""


def observation_effect(
    tool: str, arguments: dict[str, Any], payload: dict[str, Any]
) -> Effect | None:
    """What a successful proxied call did to one observation, or ``None`` (S13).

    The id comes from the result when Engram returns one (``id``, a finalize's
    ``summary_id``, a consolidation's survivor) and from the arguments for the
    tools whose result names none (``mem_update``, ``mem_get_observation``).
    """
    project = payload.get("project") if isinstance(payload.get("project"), str) else ""
    if tool in ("mem_save", "mem_session_summary"):
        found = _observation_id(payload.get("id"))
        action = "saved" if tool == "mem_save" else "session_summary"
        return Effect(found, action, saved_status(found), project) if found else None
    if tool in ("mem_update", "mem_get_observation", "mem_pin", "mem_unpin"):
        found = _observation_id(payload.get("id")) or _observation_id(arguments.get("id"))
        if not found:
            return None
        render, action = {
            "mem_update": (updated_status, "updated"),
            "mem_get_observation": (observation_status, "observation"),
            "mem_pin": (pinned_status, "pinned"),
            "mem_unpin": (unpinned_status, "unpinned"),
        }[tool]
        return Effect(found, action, render(found), project)
    if tool == "mem_consolidate" and payload.get("status") == "consolidated":
        survivor = payload.get("survivor") if isinstance(payload.get("survivor"), dict) else {}
        found = _observation_id(survivor.get("id"))
        if not found:
            return None
        owner = survivor.get("project") if isinstance(survivor.get("project"), str) else ""
        status = consolidated_status(found, payload.get("soft_deleted_ids"))
        return Effect(found, "consolidated", status, owner or project)
    if tool == "mem_session_finalize":
        found = _observation_id(payload.get("summary_id"))
        if not found:
            return None
        already = payload.get("state") == "already_finalized"
        label = "session already finalized" if already else "session finalized"
        return Effect(found, "session_summary", f"🧠 {label}, summary #{found}", project)
    return None


def with_status(payload: dict[str, Any], status: str) -> dict[str, Any]:
    """``payload`` plus recall's status line, never overwriting a key Engram sent."""
    key = "status" if "status" not in payload else "recall_status"
    return {**payload, key: status}


class McpProxy:
    """The provider's long-lived MCP child: lazy start, one call at a time, one restart."""

    def __init__(
        self,
        binary: str,
        *,
        cwd: Callable[[], str] | None = None,
        log_file: Callable[[], str] | None = None,
    ) -> None:
        self._binary = binary
        self._cwd = cwd or (lambda: "")
        self._log_file = log_file or (lambda: "")
        self._call_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._child: McpChild | None = None
        self._spawned = False
        self._restart_used = False
        self._closed = False

    def child_alive(self) -> bool:
        child = self._child
        return child is not None and child.alive()

    def call(self, name: str, arguments: dict[str, Any]) -> CallOutcome:
        """Forward one ``tools/call``. Never raises; never retries a sent call."""
        timeout = CALL_TIMEOUT_S
        # Long enough to queue behind one whole call (start, call, kill), never forever.
        wait = START_TIMEOUT_S + timeout + 2 * STOP_TIMEOUT_S
        if not self._call_lock.acquire(timeout=wait):
            return CallOutcome({"error": "Engram MCP is busy with another call; try again"})
        try:
            child, problem = self._ensure_child()
            if child is None:
                return CallOutcome({"error": problem})
            try:
                result = child.request(
                    "tools/call", {"name": name, "arguments": arguments}, timeout
                )
            except McpRefusal as exc:
                return CallOutcome({"error": f"Engram MCP refused {name}: {exc}"}, refused=True)
            except McpError as exc:
                self._discard(child)
                if exc.timed_out:
                    message = f"Engram MCP did not answer {name} within {timeout:g}s"
                else:
                    message = f"Engram MCP exited during {name}"
                if exc.delivered:
                    message += "; it may have been applied, so check before repeating a write"
                return CallOutcome({"error": message}, uncertain=exc.delivered)
            payload, is_error = render_result(result)
            return CallOutcome(payload, ok=not is_error, refused=is_error)
        except Exception as exc:  # noqa: BLE001 - the contract is "never raise"
            logger.debug("recall: MCP call %s failed", name, exc_info=True)
            return CallOutcome({"error": f"{name} failed: {type(exc).__name__}"})
        finally:
            self._call_lock.release()

    def _ensure_child(self) -> tuple[McpChild | None, str]:
        """The running child, starting it (or its one restart) when needed."""
        with self._state_lock:
            if self._closed:
                return None, "Engram MCP proxy is stopped"
            child = self._child
        if child is not None and child.alive():
            return child, ""
        if child is not None:
            self._discard(child)
        if self._spawned:
            if self._restart_used:
                return None, (
                    "Engram MCP is not running and was already restarted once in this "
                    "session; memory tools are unavailable until Hermes restarts"
                )
            self._restart_used = True
        self._spawned = True
        try:
            child = McpChild.spawn(self._binary, cwd=self._cwd(), log_path=self._log_file())
        except McpError as exc:
            return None, f"could not start Engram MCP: {exc}"
        try:
            child.initialize(START_TIMEOUT_S)
        except (McpError, McpRefusal) as exc:
            child.stop()
            return None, f"could not start Engram MCP: {exc}"
        with self._state_lock:
            if self._closed:
                child.stop()
                return None, "Engram MCP proxy is stopped"
            self._child = child
        return child, ""

    def _discard(self, child: McpChild) -> None:
        with self._state_lock:
            if self._child is child:
                self._child = None
        child.kill()
        child.stop()

    def close(self) -> None:
        """Stop the child without waiting for an in-flight call. Never raises."""
        try:
            with self._state_lock:
                self._closed = True
                child = self._child
                self._child = None
            if child is not None:
                child.stop()
        except Exception:
            logger.debug("recall: MCP proxy close failed", exc_info=True)
