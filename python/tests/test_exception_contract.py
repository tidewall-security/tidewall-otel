"""Is `except tidewall_otel.TidewallError` TOTAL over every declination?

The claim the README makes -- "one `except` clause covers every way the agent
can decline a call" -- is only worth making if it holds for every way. This
module enumerates them and drives each one end to end through a real activated
client.

It matters because `GuardAPIError` and its subclasses in `_http.py` are
deliberately OUTSIDE the `TidewallError` hierarchy. They are internal transport
failures and are meant to be wrapped before they reach a caller. "Meant to be"
is a claim; these tests are the evidence. An unexpected exception type
propagating out of a security agent is a fail-open in the CALLER's error
handling: the application had a handler, and the agent walked past it.
"""

import httpx
import openai
import pytest

import tidewall_otel
import tidewall_otel._guard as G
from tidewall_otel._http import GuardAPIError, GuardSchemaInvalid, GuardTimeout, GuardUnreachable

_OK = {"id": "x", "object": "chat.completion", "created": 0, "model": "gpt-4o",
       "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"}}]}
_CLEAN = {"result": {"blocked": False, "transformed": False, "policy": "p"}}


@pytest.fixture(autouse=True)
def _clean_activation_state():
    """Each test activates, so each must start deactivated.

    Without this, `activate()` returns early with "already active" and never
    validates configuration -- so the invalid-config test passed no config
    check at all and simply failed to raise. A test that cannot reach the
    behaviour it names is worth nothing.
    """
    tidewall_otel.deactivate()
    yield
    tidewall_otel.deactivate()


def _client():
    return openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_OK))))


def _raises(exc):
    def _guard(**kwargs):
        raise exc
    return _guard


#: (label, guard behaviour, extra call kwargs, expected concrete type)
_DECLINATIONS = [
    ("blocked verdict",
     lambda **kw: {"result": {"blocked": True, "transformed": False, "policy": "p"}},
     {}, tidewall_otel.TidewallBlockedError),
    ("lossy input",
     lambda **kw: _CLEAN,
     {"extra_body": {"messages": [{"role": "user", "content": "EVIL"}]}},
     tidewall_otel.LossyInputError),
    ("guard unreachable", _raises(GuardUnreachable("refused")), {},
     tidewall_otel.TidewallRefusedError),
    ("guard timeout", _raises(GuardTimeout("timed out")), {},
     tidewall_otel.TidewallRefusedError),
    ("guard schema invalid", _raises(GuardSchemaInvalid("bad body")), {},
     tidewall_otel.TidewallRefusedError),
    ("malformed guard response", lambda **kw: {"result": {"nonsense": True}}, {},
     tidewall_otel.TidewallRefusedError),
    ("bare GuardAPIError", _raises(GuardAPIError("raw")), {},
     tidewall_otel.TidewallRefusedError),
]


@pytest.mark.parametrize("label,guard,call_kwargs,expected",
                         _DECLINATIONS, ids=[d[0] for d in _DECLINATIONS])
def test_every_declination_is_a_TidewallError(monkeypatch, label, guard,
                                              call_kwargs, expected):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setattr(G, "post_guard", guard)

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallError) as caught:
        _client().chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            **call_kwargs)

    assert isinstance(caught.value, expected), (
        f"{label}: got {type(caught.value).__name__}, expected "
        f"{expected.__name__}")

    # And the CONCRETE type must be publicly nameable. Catching it via the
    # base class is not the whole contract: a caller who wants to tell "you
    # sent something I cannot inspect" (fixable -- drop `extra_body`) from
    # "the guard failed" (not fixable) has to name the narrower type, and
    # cannot if it lives only in a private module. `LossyInputError` was
    # exactly that case; nothing failed when it was un-exported, because
    # every test caught it through `TidewallError`.
    concrete = type(caught.value).__name__
    assert concrete in tidewall_otel.__all__, (
        f"{label}: callers receive {concrete}, which is not in __all__")


def test_a_transport_failure_NEVER_escapes_as_a_GuardAPIError(monkeypatch):
    """The specific hazard, stated as its own test.

    `GuardAPIError` is not a `TidewallError`. If one reached application code,
    every handler written against the documented contract would miss it.
    """
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setattr(G, "post_guard", _raises(GuardUnreachable("refused")))

    tidewall_otel.activate()
    try:
        _client().chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    except BaseException as exc:          # noqa: BLE001 -- that IS the assertion
        assert not isinstance(exc, GuardAPIError), (
            f"a raw {type(exc).__name__} reached the caller")
        assert isinstance(exc, tidewall_otel.TidewallError)
    else:
        pytest.fail("an unreachable guard did not decline the call in enforce")


def test_the_BLOCK_activation_policy_also_raises_a_TidewallError(monkeypatch):
    """Raised for EVERY call under that policy, so it is the one an
    application is most likely to meet. It subclassed a bare `RuntimeError`."""
    monkeypatch.setenv("TIDEWALL_BASE_URL", "")
    monkeypatch.setenv("TIDEWALL_TOKEN", "")
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "block")

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallError):
        openai.OpenAI(api_key="t").chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])


def test_invalid_CONFIGURATION_raises_a_TidewallError(monkeypatch):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "")
    monkeypatch.setenv("TIDEWALL_TOKEN", "")
    monkeypatch.delenv("TIDEWALL_ON_ACTIVATION_FAILURE", raising=False)

    with pytest.raises(tidewall_otel.TidewallError):
        tidewall_otel.activate()
