"""A programmable in-process Engram stand-in, so tests never touch a real server.

The user's own Engram runs on 127.0.0.1:7437 with their real memories in it. No
test may reach it, so :class:`FakeEngram` binds port 0 (the OS picks a free one)
and every test points the client at ``base_url``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any
from urllib.parse import parse_qs, urlparse

# A route's value is either a ready-made ``(status, payload)`` or a callable that
# receives the recorded request and returns one.
Response = tuple[int, Any]
Route = Response | Callable[["RecordedRequest"], Response]


@dataclass
class RecordedRequest:
    """One request as the fake server saw it."""

    method: str
    path: str
    query: dict[str, list[str]] = field(default_factory=dict)
    json_body: Any = None
    raw_body: bytes = b""
    authorization: str | None = None

    def query_one(self, key: str) -> str | None:
        """First value for ``key``, or ``None`` when the param was not sent."""
        values = self.query.get(key)
        return values[0] if values else None


class Delay:
    """A route marker that sleeps before answering, to exercise client timeouts."""

    def __init__(self, seconds: float, response: Response = (200, {"status": "ok"})) -> None:
        self.seconds = seconds
        self.response = response


class Malformed:
    """A route marker that answers 2xx with bytes that are not JSON."""

    def __init__(self, status: int = 200, raw: bytes = b"<html>not json</html>") -> None:
        self.status = status
        self.raw = raw


class FakeEngram:
    """A threaded HTTP server with programmable routes and a request log."""

    def __init__(
        self,
        routes: dict[tuple[str, str], Route] | None = None,
        default_response: Response = (404, {"error": "not found"}),
        port: int = 0,
    ) -> None:
        self.routes: dict[tuple[str, str], Route] = dict(routes or {})
        self.default_response: Response = default_response
        self.requests: list[RecordedRequest] = []
        self._lock = threading.Lock()

        # ``port`` is 0 (the OS picks) except for tests that need a server to come
        # up on a port a client was already pointed at while nothing listened there.
        server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(self))
        server.daemon_threads = True
        self._server = server
        # A short poll interval matters: BaseServer.shutdown() blocks until the
        # serve_forever loop next wakes up, so the 0.5s default would add half a
        # second to every single fixture teardown.
        self._thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self._thread.start()

    # -- Lifecycle -----------------------------------------------------------

    @property
    def base_url(self) -> str:
        """The server's base URL, with no trailing slash."""
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def port(self) -> int:
        """The port the OS assigned."""
        return int(self._server.server_address[1])

    def stop(self) -> None:
        """Shut the server down and join its thread."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> FakeEngram:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # -- Programming ---------------------------------------------------------

    def route(self, method: str, path: str, response: Route) -> None:
        """Register (or replace) the handler for one ``(method, path)`` pair."""
        self.routes[(method.upper(), path)] = response

    def reset(self) -> None:
        """Forget every recorded request."""
        with self._lock:
            self.requests.clear()

    # -- Inspection ----------------------------------------------------------

    @property
    def last_request(self) -> RecordedRequest | None:
        """The most recently recorded request, or ``None`` when none arrived."""
        with self._lock:
            return self.requests[-1] if self.requests else None

    def find(self, method: str, path: str) -> RecordedRequest | None:
        """The first recorded request matching ``method`` and ``path``."""
        with self._lock:
            for request in self.requests:
                if request.method == method.upper() and request.path == path:
                    return request
        return None

    # -- Internals -----------------------------------------------------------

    def _record(self, request: RecordedRequest) -> None:
        with self._lock:
            self.requests.append(request)

    def _resolve(self, request: RecordedRequest) -> Route:
        return self.routes.get((request.method, request.path), self.default_response)


def _make_handler(fake: FakeEngram) -> type[BaseHTTPRequestHandler]:
    """Build a handler class bound to one :class:`FakeEngram` instance."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            """Silence the stderr access log so pytest output stays readable."""

        def _handle(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                parsed_body = json.loads(raw.decode("utf-8")) if raw else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed_body = None

            request = RecordedRequest(
                method=self.command,
                path=parsed.path,
                query=parse_qs(parsed.query, keep_blank_values=True),
                json_body=parsed_body,
                raw_body=raw,
                authorization=self.headers.get("Authorization"),
            )
            fake._record(request)

            route = fake._resolve(request)
            if isinstance(route, Delay):
                time.sleep(route.seconds)
                route = route.response
            if isinstance(route, Malformed):
                self._send_raw(route.status, route.raw, "text/html")
                return
            if callable(route):
                route = route(request)
                if isinstance(route, Malformed):
                    self._send_raw(route.status, route.raw, "text/html")
                    return

            status, payload = route
            body = b"" if payload is None else json.dumps(payload).encode("utf-8")
            self._send_raw(status, body, "application/json")

        def _send_raw(self, status: int, body: bytes, content_type: str) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The client gave up first (timeout tests do exactly this).
                pass

        do_GET = _handle
        do_POST = _handle
        do_PATCH = _handle
        do_PUT = _handle
        do_DELETE = _handle

    return Handler
