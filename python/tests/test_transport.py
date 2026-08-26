"""Transport hardening: TLS, redirects, and typed failures."""

import io
import urllib.error
import urllib.request

import pytest

from tidewall_otel._http import GuardAPIError, post_guard

ENDPOINT = "/v1/guard_chat_completions"


def _response(status, headers=None, body=b""):
    """An object with .status, .read(n) and .headers.

    Local to this module: it is an HTTP-response double, unrelated to the
    guard-response factories used elsewhere, and two different helpers must
    not share a name.
    """

    class _Resp(io.BytesIO):
        def __init__(self):
            super().__init__(body)
            self.status = status
            self.code = status
            self.headers = headers or {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Resp()


class RecordingOpener:
    """Records every (destination, headers, body) it is asked to open.

    Recording at the OPENER, not the socket: under HTTPS the bearer token is
    encrypted before any socket write, so captured wire bytes cannot show it,
    and a byte string does not identify its destination.
    """

    def __init__(self, response=None):
        self.opened: list[tuple[str, dict, bytes]] = []
        self._response = response

    def open(self, request, timeout=None):
        # Headers too: a recording that drops them cannot show whether the
        # credential was sent, only that a URL was visited.
        self.opened.append(
            (request.full_url, dict(request.headers), request.data or b"")
        )
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_a_non_https_url_is_refused_before_any_connection():
    opener = RecordingOpener()
    with pytest.raises(GuardAPIError, match="https"):
        post_guard(base_url="http://guard.example", token="secret",
                   payload={}, opener=opener)
    assert opener.opened == [], "a connection was attempted to a rejected URL"


def test_a_redirect_is_refused_and_nothing_reaches_the_target():
    """One request to the original origin, and NO request to the target."""
    opener = RecordingOpener(
        response=_response(302, {"Location": "https://elsewhere.example"})
    )
    with pytest.raises(GuardAPIError, match="redirect"):
        post_guard(base_url="https://guard.example", token="secret",
                   payload={"p": 1}, opener=opener)

    destinations = [url for url, _headers, _body in opener.opened]
    assert destinations == ["https://guard.example" + ENDPOINT]
    assert not any("elsewhere.example" in url for url in destinations)


def test_the_credential_and_payload_reach_the_validated_origin():
    """Asserts the credential and prompt WERE sent, and where.

    Recording only URL and body and asserting a URL prefix would pass if
    production dropped the credential and payload entirely. A test that a
    leak-free implementation and a send-nothing implementation both satisfy
    is not a test of where bytes go.
    """
    opener = RecordingOpener(response=_response(200, body=b'{"ok": true}'))
    post_guard(base_url="https://guard.example", token="secret",
               payload={"prompt": "CANARY"}, opener=opener)

    assert len(opener.opened) == 1
    url, headers, body = opener.opened[0]
    assert url == "https://guard.example" + ENDPOINT
    assert headers["Authorization"] == "Bearer secret"
    assert b"CANARY" in body


def test_the_production_opener_refuses_a_real_redirect():
    """The injected RecordingOpener never exercises production's opener, so
    the tests above pass even if production still follows redirects."""
    from tidewall_otel._http import _build_opener

    opener = _build_opener()
    handlers = [type(h).__name__ for h in opener.handlers]
    assert "_RefuseRedirects" in handlers, handlers

    handler = next(h for h in opener.handlers if type(h).__name__ == "_RefuseRedirects")
    request = urllib.request.Request("https://guard.example" + ENDPOINT)
    with pytest.raises(GuardAPIError, match="refusing redirect"):
        handler.redirect_request(
            request, io.BytesIO(), 302, "Found", {},
            "https://elsewhere.example",
        )


@pytest.mark.parametrize("status", [200, 500])
def test_an_oversized_body_is_rejected_not_truncated(status):
    """Both paths. An error path calling exc.read() unbounded and only then
    slicing has already read the whole body into memory."""
    oversized = b"x" * (1024 * 1024 + 1)
    if status == 200:
        opener = RecordingOpener(response=_response(200, body=oversized))
    else:
        opener = RecordingOpener(response=urllib.error.HTTPError(
            "https://guard.example" + ENDPOINT, 500, "err", {},
            io.BytesIO(oversized),
        ))
    with pytest.raises(GuardAPIError, match="too large"):
        post_guard(base_url="https://guard.example", token="t",
                   payload={}, opener=opener)


def test_the_socket_timeout_parameter_is_NOT_called_timeout():
    """Task 3 renames it: the outer caller deadline belongs to the executor,
    and a per-connection read bound is a different thing needing a different
    name. Every later caller must be updated in the same breath."""
    import inspect

    params = inspect.signature(post_guard).parameters
    assert "socket_timeout" in params
    assert "timeout" not in params, "the renamed parameter still exists"


# -- failure classification ------------------------------------------------
# The injected-opener tests never raise, so they leave the typed-exception
# mapping unconstrained -- every `raise Guard*` in the error paths needs a
# case that actually reaches it.

@pytest.mark.parametrize("raised,expected", [
    (urllib.error.URLError(TimeoutError("timed out")), "GuardTimeout"),
    (urllib.error.URLError(ConnectionRefusedError("refused")), "GuardUnreachable"),
    (urllib.error.URLError(OSError("dns")), "GuardUnreachable"),
    (TimeoutError("socket deadline"), "GuardTimeout"),
])
def test_each_transport_failure_maps_to_its_TYPED_exception(raised, expected):
    """Dispatch maps these to outcomes, so collapsing them to a bare
    GuardAPIError would file a timeout as invariant_violated -- a policy
    decision reported as a bug."""
    from tidewall_otel import _http

    opener = RecordingOpener(response=raised)
    with pytest.raises(getattr(_http, expected)):
        post_guard(base_url="https://guard.example", token="t",
                   payload={}, opener=opener)


def test_a_non_json_body_is_SCHEMA_INVALID_not_unreachable():
    """The guard answered; it answered wrongly. Reporting unreachable would
    send the caller chasing a network problem that does not exist."""
    from tidewall_otel._http import GuardSchemaInvalid

    opener = RecordingOpener(response=_response(200, body=b"<html>nope</html>"))
    with pytest.raises(GuardSchemaInvalid):
        post_guard(base_url="https://guard.example", token="t",
                   payload={}, opener=opener)


def test_an_empty_body_is_an_empty_dict_not_an_error():
    """A 200 with no body is a degenerate but well-formed answer; the response
    classifier decides what it means, not the transport."""
    opener = RecordingOpener(response=_response(200, body=b""))
    assert post_guard(base_url="https://guard.example", token="t",
                      payload={}, opener=opener) == {}
