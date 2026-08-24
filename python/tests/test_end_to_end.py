"""End-to-end through a REAL activated client.

THE TEST CLASS THIS SUITE WAS MISSING. Every dispatch test called
dispatch_sync directly, so they proved the component works and never that
activation wires it. The result was 391 passing tests over an agent that
refused every call in enforce and passed every call unguarded in monitor,
because the factories were handed no executor and the AttributeError was
filed as an internal invariant violation.

These tests go through `tidewall_otel.activate()` and a real provider client.
"""

import json

import httpx
import pytest

import openai

import tidewall_otel
from tidewall_otel._exceptions import TidewallError

#: One well-formed chat completion, reused wherever a provider must answer.
_COMPLETION = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "gpt-4o",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "ok"}}],
}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    import os

    for name in list(os.environ):
        if name.startswith("TIDEWALL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    yield
    tidewall_otel.deactivate()


@pytest.fixture
def guard_says(monkeypatch):
    """Replace the transport, not the guard: everything from _payload_for
    downwards stays real, so a wiring error between them is still visible."""
    asked = []

    def install(result):
        def fake_post(**kwargs):
            asked.append(kwargs["payload"])
            return {"request_id": "r", "request_time": "t", "summary": "",
                    "result": result}

        monkeypatch.setattr("tidewall_otel._guard.post_guard", fake_post)
        return asked

    return install


@pytest.fixture
def provider():
    """A real OpenAI client whose transport records the outgoing body."""
    reached = []

    def handler(request):
        reached.append(json.loads(request.content))
        return httpx.Response(200, json=_COMPLETION)

    client = openai.OpenAI(
        api_key="t", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, reached


CLEAN = {"blocked": False, "transformed": False, "policy": "d"}
BLOCKED = {"blocked": True, "transformed": False, "policy": "d"}
TRANSFORMED = {"blocked": False, "transformed": True, "policy": "d",
               "guard_output": {"messages": [{"role": "user", "content": "REDACTED"}]}}


def test_a_clean_call_REACHES_the_guard_and_then_the_provider(guard_says, provider):
    """The base case, and the one that was broken: the guard must actually be
    contacted through an activated client."""
    asked = guard_says(CLEAN)
    client, reached = provider

    tidewall_otel.activate()
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert len(asked) == 1, "the guard was never contacted through activation"
    assert asked[0]["guard_input"]["messages"][0]["content"] == "hi"
    assert len(reached) == 1, "the provider was not reached"


def test_a_BLOCKED_verdict_stops_an_activated_call(guard_says, provider):
    asked = guard_says(BLOCKED)
    client, reached = provider

    tidewall_otel.activate()
    with pytest.raises(TidewallError):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert len(asked) == 1
    assert reached == [], "a blocked call reached the provider"


def test_a_TRANSFORM_rewrites_what_the_provider_receives(guard_says, provider):
    guard_says(TRANSFORMED)
    client, reached = provider

    tidewall_otel.activate()
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "SECRET"}])

    assert reached[0]["messages"][0]["content"] == "REDACTED"
    assert not any("SECRET" in json.dumps(body) for body in reached)


def test_extra_body_is_REFUSED_through_an_activated_client(guard_says, provider):
    """P0-11 end to end: the bypass never reaches either side."""
    asked = guard_says(CLEAN)
    client, reached = provider

    tidewall_otel.activate()
    with pytest.raises(TidewallError):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "SAFE"}],
            extra_body={"messages": [{"role": "user", "content": "EVIL"}]})

    assert asked == [], "lossy input was sent to the guard"
    assert reached == [], "the bypass reached the provider"


@pytest.mark.asyncio
async def test_the_ASYNC_path_is_wired_too(guard_says, monkeypatch):
    """The async factories take the same collaborators, and nothing asserted
    they were given them either."""
    asked = guard_says(CLEAN)
    reached = []

    def handler(request):
        reached.append(json.loads(request.content))
        return httpx.Response(200, json=_COMPLETION)

    client = openai.AsyncOpenAI(
        api_key="t",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    tidewall_otel.activate()
    await client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert len(asked) == 1, "the async path never contacted the guard"
    assert len(reached) == 1


def test_MONITOR_reaches_the_provider_AND_the_guard(monkeypatch, guard_says, provider):
    """Monitor must still inspect. An unwired executor made monitor proceed
    with no guard call at all -- a silent bypass that looks like success."""
    monkeypatch.setenv("TIDEWALL_MODE", "monitor")
    asked = guard_says(BLOCKED)
    client, reached = provider

    tidewall_otel.activate()
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert len(asked) == 1, "monitor did not inspect"
    assert len(reached) == 1, "monitor did not proceed"


def test_activation_reports_a_state_that_MATCHES_reality(guard_says, provider):
    """`active=True` with a broken wiring is the worst possible answer.

    Note what this asserts and what it does NOT. Every client in this file is
    built on `httpx.MockTransport`, which IS a construction-time escape -- the
    test harness can rewrite the provider-bound body exactly as a hostile
    integrator could. So the honest report here is `unverified`, and a test
    demanding `is_active() is True` would be demanding that the escape
    detector fail. The positive direction is asserted separately, on a client
    whose transport is the standard one.
    """
    guard_says(CLEAN)
    client, _reached = provider

    tidewall_otel.activate()
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    state = tidewall_otel.state()
    assert state.lifecycle == "installed"
    assert state.surfaces["Completions.create"] == "unverified"
    assert state.is_active() is False
    assert [(e.kind, e.surface, e.reason) for e in state.events] == [
        ("unverified", "Completions.create", "client_escapes")]


def test_an_ORDINARY_client_stays_covered_and_active(guard_says, monkeypatch):
    """The other direction: escape detection must not condemn everyone.

    `httpx.Client()` with no arguments builds a standard `HTTPTransport`, so
    intercepting at `handle_request` leaves the transport TYPE untouched and
    the surface must stay `covered`. Without this test, a detector that
    returned `("transport",)` unconditionally would still pass every
    assertion above -- downgrading every honest application to `unverified`
    and making `is_active()` permanently False.
    """
    guard_says(CLEAN)
    reached = []

    def handle(self, request):
        reached.append(json.loads(request.content))
        return httpx.Response(200, json=_COMPLETION, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)

    client = openai.OpenAI(api_key="k", http_client=httpx.Client())
    tidewall_otel.activate()
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    state = tidewall_otel.state()
    assert len(reached) == 1
    assert state.surfaces["Completions.create"] == "covered"
    assert state.events == []
    assert state.is_active() is True
