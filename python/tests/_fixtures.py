"""Shared test helpers.

Task 1b of the P0 remediation plan.

DEVIATION FROM THE PLAN, DECLARED RATHER THAN SILENT
----------------------------------------------------
Task 1b specifies one upfront contract for every helper the plan uses. That
cannot be implemented in Task 1b's position: the contract references
``BoundedExecutor`` (Task 2's ``_execution.py``), ``SURFACES`` (Task 4's
``_manifest.py``) and ``dispatch_sync``/``dispatch_async`` (Task 12's
``_dispatch.py``), none of which exist yet.

So this module contains the helpers that depend only on what exists today, and
each later task adds the helpers it introduces. The plan's *rule* is unchanged
and is what actually matters: a test module imports exactly the helpers it
uses, and ``pyflakes`` enforces that nothing is called undefined.

A single upfront fixture module is the wrong shape for a plan whose tasks
create the very objects the fixtures wrap. It must either forward-reference
everything or come last.
"""

from __future__ import annotations

import http.server
import json
import subprocess
import sys
import threading
import time
import typing
from pathlib import Path

# Capacity constants, shared by the executor fixture and the boundary tests so
# the two cannot disagree about what "full" means.
MAX_WORKERS = 4
QUEUE_SIZE = 32


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    """Block until ``predicate()`` is true, or fail loudly.

    Used wherever work drains asynchronously; a bare ``sleep`` would either
    flake or slow the suite.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def _sdk_installed(module: str) -> bool:
    """Whether a provider SDK is importable.

    Both are optional extras, so every test touching a real signature must
    skip rather than error when its SDK is absent.
    """
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _admits_free_form(annotation, _depth: int = 0, _seen=None) -> str | None:
    """A reason string if ``annotation`` transitively admits an unconstrained
    value -- bare ``object``/``Any``, or a mapping to either -- else ``None``.

    Callers MUST resolve annotations first, via
    ``inspect.signature(..., eval_str=True)``. Both provider SDKs use
    ``from __future__ import annotations``, so unresolved annotations are
    plain strings and this returns ``None`` for absolutely everything -- so a
    scan that forgets `eval_str` reports a clean bill of health for every
    parameter on both providers.
    """
    if _depth > 5:
        return None
    _seen = set() if _seen is None else _seen

    if annotation in (object, typing.Any):
        return "bare object/Any"

    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (dict, typing.Mapping) or getattr(origin, "__name__", "") == "Mapping":
        if len(args) == 2 and args[1] in (object, typing.Any):
            return f"{getattr(origin, '__name__', origin)}[..., object]"
    for arg in args:
        reason = _admits_free_form(arg, _depth + 1, _seen)
        if reason:
            return reason

    if hasattr(annotation, "__annotations__") and hasattr(annotation, "__required_keys__"):
        name = getattr(annotation, "__name__", str(annotation))
        if name in _seen:
            return None
        _seen.add(name)
        try:
            hints = typing.get_type_hints(annotation)
        except Exception:
            hints = {}
        for field, field_annotation in hints.items():
            reason = _admits_free_form(field_annotation, _depth + 1, _seen)
            if reason:
                return f"{name}.{field} -> {reason}"
    return None


def run_python(*args: str, env: dict | None = None) -> str:
    """Run a subprocess Python and return stdout.

    Import side effects cannot be undone in-process, so anything asserting
    what was or was not imported must run out of process.
    """
    result = subprocess.run([sys.executable, *args], capture_output=True,
                            text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def undefined_names_under(root) -> list[str]:
    """Every ``undefined name`` pyflakes reports beneath ``root``.

    Delegated rather than hand-rolled: correct scope analysis is a hard
    problem, and earlier bespoke versions of this check missed attribute
    calls, chained attributes, and -- worst -- a function parameter binding a
    name module-wide and thereby excusing genuinely undefined calls elsewhere
    in the same file.

    KNOWN LIMITATION: a name bound only by a walrus on an unreachable path is
    not reported. That is an ``UnboundLocalError`` at runtime rather than a
    name-resolution defect, and pyflakes does not model control flow.
    """
    result = subprocess.run([sys.executable, "-m", "pyflakes", str(root)],
                            capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if "undefined name" in line]


class _GuardHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:                      # noqa: N802 (stdlib name)
        length = int(self.headers.get("Content-Length", 0))
        self.server.requests.append(json.loads(self.rfile.read(length) or b"{}"))
        body = json.dumps(self.server.response_body).encode()
        self.send_response(self.server.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:           # silence per-request logs
        pass


class LocalGuardServer:
    """A real HTTP server on localhost.

    Real, not a mock, because the transport is ``urllib.request.urlopen``:
    httpx-based interception (respx) cannot see it at all.
    """

    def __init__(self) -> None:
        self._server = http.server.HTTPServer(("127.0.0.1", 0), _GuardHandler)
        self._server.requests = []
        self._server.response_body = {}
        self._server.response_status = 200
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def requests(self) -> list:
        return self._server.requests

    def respond(self, *, json: dict | None = None, status: int = 200) -> None:
        self._server.response_body = json or {}
        self._server.response_status = status

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
