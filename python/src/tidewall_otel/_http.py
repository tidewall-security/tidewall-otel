"""Minimal HTTP client for the Tidewall guard API.

Implemented inline so this package has no hard dependency on any SDK.
Only the standard library ``urllib`` is used — no requests, no httpx,
no extra wheels in the dependency tree.

Talks to a Tidewall guard server (or any compatible implementation of the
``/v1/guard_chat_completions`` endpoint). The request and response shapes
match the Tidewall API contract.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger("tidewall.otel.http")

#: Responses larger than this are refused outright rather than truncated. A
#: guard verdict is small; anything larger is a misconfiguration or an attempt
#: to exhaust memory, and truncating would leave us parsing a prefix.
MAX_RESPONSE_BYTES = 1024 * 1024


class GuardAPIError(Exception):
    """Raised when the guard API returns a non-2xx response or fails to respond.

    The original network or parsing error is chained via ``__cause__``.
    """


class GuardUnreachable(GuardAPIError):
    """Connection refused, DNS failure, or TLS failure."""


class GuardTimeout(GuardAPIError):
    """The socket deadline elapsed."""


class GuardSchemaInvalid(GuardAPIError):
    """The response was not JSON, or not the expected shape."""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect.

    urllib follows redirects by default and re-sends the Authorization header
    to the new location, so a compromised or misconfigured guard URL can
    forward the bearer token and the prompt to a third party. There is no
    legitimate reason for the guard endpoint to redirect.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GuardAPIError(f"refusing redirect {code} to {newurl}")


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_RefuseRedirects)


def _read_bounded(stream, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    """Read at most ``limit`` bytes, refusing anything larger.

    Reads limit+1 and rejects on overflow: reading the whole body and then
    slicing has already consumed the memory the bound exists to protect.
    """
    raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise GuardAPIError(f"guard response too large (> {limit} bytes)")
    return raw


def post_guard(
    *,
    base_url: str,
    token: str,
    payload: dict[str, Any],
    socket_timeout: float = 10.0,
    opener: urllib.request.OpenerDirector | None = None,
) -> dict[str, Any]:
    """POST a request to ``/v1/guard_chat_completions`` and return the parsed JSON.

    Args:
        base_url: Root URL of the guard server. Trailing slashes are tolerated.
        token: Bearer token for the ``Authorization`` header.
        payload: Request body — must already match the Tidewall guard schema.
        socket_timeout: Per-connection read bound in seconds. NOT a bound on
            caller latency -- that belongs to the BoundedExecutor wrapping
            this call, and the two are different things with different names.
        opener: Injected in tests. Production uses the redirect-refusing
            opener built by ``_build_opener``.

    Returns:
        The parsed JSON response body as a dict.

    Raises:
        GuardAPIError: For HTTP errors, network failures, or invalid JSON.
    """
    url = base_url.rstrip("/") + "/v1/guard_chat_completions"

    # Scheme validation happens HERE, before any connection is attempted, and
    # in this function rather than in config: the tests call post_guard
    # directly and expect it to refuse.
    scheme = urllib.parse.urlparse(url).scheme
    if scheme != "https":
        raise GuardAPIError(
            f"refusing to send the bearer token and prompt over {scheme!r}; "
            f"the guard URL must use https"
        )

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "tidewall-otel/0.1.0",
        },
    )

    opener = opener if opener is not None else _build_opener()

    try:
        with opener.open(request, timeout=socket_timeout) as response:
            # DEFENCE IN DEPTH. `_RefuseRedirects` stops urllib following a
            # redirect, but that handler lives in the opener -- and the opener
            # is injectable. If any opener hands back a 3xx, refuse it here
            # rather than parsing a redirect body as a guard verdict.
            status = getattr(response, "status", None) or getattr(response, "code", None)
            if status is not None and 300 <= status < 400:
                location = ""
                try:
                    location = response.headers.get("Location", "")
                except Exception:
                    pass
                raise GuardAPIError(f"refusing redirect {status} to {location}")

            raw = _read_bounded(response).decode("utf-8")
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        # BOUNDED on the error path too. Calling exc.read() unbounded and
        # then slicing has already read the whole body into memory, which is
        # exactly what the bound exists to prevent.
        error_body = ""
        try:
            error_body = _read_bounded(exc).decode("utf-8", errors="replace")[:500]
        except GuardAPIError:
            raise
        except Exception:
            pass
        raise GuardUnreachable(
            f"Guard API returned HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        # A socket timeout surfaces as URLError(reason=TimeoutError). Typed
        # subclasses of GuardAPIError so existing `except GuardAPIError`
        # callers keep working while dispatch can map precisely.
        if isinstance(exc.reason, TimeoutError):
            raise GuardTimeout(f"Guard API timed out: {exc.reason}") from exc
        raise GuardUnreachable(f"Guard API unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise GuardTimeout(f"Guard API timed out: {exc}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise GuardSchemaInvalid(f"Guard API returned invalid JSON: {exc}") from exc
