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
import urllib.request
from typing import Any

logger = logging.getLogger("tidewall.otel.http")


class GuardAPIError(Exception):
    """Raised when the guard API returns a non-2xx response or fails to respond.

    The original network or parsing error is chained via ``__cause__``.
    """


def post_guard(
    *,
    base_url: str,
    token: str,
    payload: dict[str, Any],
    timeout: float = 10.0,
) -> dict[str, Any]:
    """POST a request to ``/v1/guard_chat_completions`` and return the parsed JSON.

    Args:
        base_url: Root URL of the guard server. Trailing slashes are tolerated.
        token: Bearer token for the ``Authorization`` header.
        payload: Request body — must already match the Tidewall guard schema.
        timeout: Per-request socket timeout in seconds.

    Returns:
        The parsed JSON response body as a dict.

    Raises:
        GuardAPIError: For HTTP errors, network failures, or invalid JSON.
    """
    url = base_url.rstrip("/") + "/v1/guard_chat_completions"
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

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        # Surface the server-provided error body when present — useful for
        # debugging policy or auth issues — but cap the length to avoid
        # logging entire HTML error pages.
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise GuardAPIError(
            f"Guard API returned HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GuardAPIError(f"Guard API unreachable: {exc.reason}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise GuardAPIError(f"Guard API returned invalid JSON: {exc}") from exc
