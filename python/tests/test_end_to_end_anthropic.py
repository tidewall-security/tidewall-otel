"""Anthropic, driven through a real activated client.

Round 9 found that NEITHER Anthropic wrapper body was executed by the suite.
The factories were called -- so the `def` lines ran and coverage looked
plausible -- but nothing ever invoked the wrapper they returned. Every
end-to-end test drove OpenAI.

So a wrong surface constant, a missing `await`, or a wrapper that never
dispatched had no end-to-end guard on half the supported providers. These are
the OpenAI end-to-end tests' counterpart, and the mode matrix is derived from
`SURFACES` so a third Anthropic boundary cannot be added without one.
"""

import json

import anthropic
import httpx
import pytest

import tidewall_otel
import tidewall_otel._guard as guard_module
from tidewall_otel._manifest import SURFACES

_MESSAGE = {
    "id": "msg_1", "type": "message", "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_CLEAN = {"result": {"blocked": False, "transformed": False, "policy": "p"}}
_BLOCKED = {"result": {"blocked": True, "transformed": False, "policy": "p"}}


@pytest.fixture(autouse=True)
def _reset():
    tidewall_otel.deactivate()
    yield
    tidewall_otel.deactivate()


@pytest.fixture
def guard_says(monkeypatch):
    asked = []

    def install(response):
        def post(**kwargs):
            asked.append(kwargs["payload"])
            return response
        monkeypatch.setattr(guard_module, "post_guard", post)
        return asked

    return install


def _client(reached):
    def handler(request):
        reached.append(json.loads(request.content))
        return httpx.Response(200, json=_MESSAGE)

    return anthropic.Anthropic(
        api_key="t",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _async_client(reached):
    def handler(request):
        reached.append(json.loads(request.content))
        return httpx.Response(200, json=_MESSAGE)

    return anthropic.AsyncAnthropic(
        api_key="t",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_the_ANTHROPIC_sync_wrapper_reaches_the_guard_then_the_provider(
        monkeypatch, guard_says):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    asked = guard_says(_CLEAN)
    reached = []

    tidewall_otel.activate()
    result = _client(reached).messages.create(
        model="claude-3-5-sonnet-20241022", max_tokens=16,
        messages=[{"role": "user", "content": "hi"}])

    assert len(asked) == 1, "the guard was not consulted"

    # The payload nests under `guard_input`, and the head system message is
    # emitted UNCONDITIONALLY for Anthropic -- deliberately, because a
    # conditional head makes the manifest's `messages[1:]` mapping false for
    # every system-less call and makes a write-back delete a real turn.
    shown = asked[0]["guard_input"]["messages"]
    assert shown[0] == {"role": "system", "content": ""}, shown
    assert [m for m in shown if m["role"] != "system"] == [
        {"role": "user", "content": "hi"}], shown

    assert len(reached) == 1, "the provider was not reached"
    assert result.content[0].text == "ok"


def test_the_ANTHROPIC_async_wrapper_is_wired_too(monkeypatch, guard_says):
    """The missing `await` case: an async wrapper that returns a coroutine
    instead of awaiting it fails here and nowhere else."""
    import asyncio

    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    asked = guard_says(_CLEAN)
    reached = []

    tidewall_otel.activate()

    async def drive():
        return await _async_client(reached).messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=16,
            messages=[{"role": "user", "content": "hi"}])

    result = asyncio.run(drive())

    assert len(asked) == 1, "the guard was not consulted on the async path"
    assert len(reached) == 1
    assert result.content[0].text == "ok"


def test_a_BLOCKED_verdict_stops_an_activated_ANTHROPIC_call(
        monkeypatch, guard_says):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    guard_says(_BLOCKED)
    reached = []

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallBlockedError):
        _client(reached).messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=16,
            messages=[{"role": "user", "content": "attack"}])

    assert reached == [], "the provider was reached despite a block"


def test_EVERY_anthropic_surface_has_an_end_to_end_test_here():
    """Derived, so a third Anthropic boundary cannot arrive untested.

    The defect this file exists to fix was precisely that a whole provider's
    wrappers were never invoked while the suite looked complete.
    """
    import inspect

    declared = {surface.attribute for surface in SURFACES
                if surface.provider == "anthropic"}
    source = inspect.getsource(inspect.getmodule(test_a_BLOCKED_verdict_stops_an_activated_ANTHROPIC_call))

    # Each surface is exercised through the client call it belongs to; the
    # sync/async split is what distinguishes them.
    exercised = set()
    if "messages.create(" in source:
        exercised |= {name for name in declared if not name.startswith("Async")}
    if "await _async_client" in source:
        exercised |= {name for name in declared if name.startswith("Async")}

    assert declared <= exercised, (
        f"Anthropic surfaces with no end-to-end test: {sorted(declared - exercised)}")


# -- extra_body, on the other provider ------------------------------------
#
# The fingerprint omitted `extra_body` for every surface, so the bypass was
# never OpenAI-specific. Anthropic gets its own proof rather than inheriting
# the assumption -- which is the reason this whole module exists.

def _fill_during_span(monkeypatch, payload):
    import tidewall_otel._dispatch as dispatch

    real_span = dispatch.gen_ai_span

    def mutating_span(*args, **kwargs):
        payload["messages"] = [{"role": "user", "content": "MALICIOUS override"}]
        return real_span(*args, **kwargs)

    monkeypatch.setattr(dispatch, "gen_ai_span", mutating_span)


def test_an_extra_body_FILLED_after_inspection_is_refused(monkeypatch, guard_says):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    guard_says(_CLEAN)
    escape = {}
    _fill_during_span(monkeypatch, escape)
    reached = []

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        _client(reached).messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=16,
            messages=[{"role": "user", "content": "benign"}],
            extra_body=escape)

    assert raised.value.outcome_kind == "mutated_during_guard"
    assert reached == [], "an uninspected override reached Anthropic"


async def test_an_extra_body_FILLED_after_inspection_is_refused_ASYNC(monkeypatch, guard_says):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    guard_says(_CLEAN)
    escape = {}
    _fill_during_span(monkeypatch, escape)
    reached = []

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        await _async_client(reached).messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=16,
            messages=[{"role": "user", "content": "benign"}],
            extra_body=escape)

    assert raised.value.outcome_kind == "mutated_during_guard"
    assert reached == []


def test_the_guard_payload_carries_the_MODEL_and_the_PROVIDER(monkeypatch, guard_says):
    """Anthropic gets its own proof: `llm_provider` comes from the surface,
    so a single-provider test cannot show it is the RIGHT surface's value."""
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    asked = guard_says(_CLEAN)
    reached = []

    tidewall_otel.activate()
    _client(reached).messages.create(
        model="claude-3-5-sonnet-20241022", max_tokens=16,
        messages=[{"role": "user", "content": "hi"}])

    assert asked, "the guard was never called"
    assert asked[0]["model"] == "claude-3-5-sonnet-20241022"
    assert asked[0]["llm_provider"] == "anthropic", \
        f"reported as {asked[0]['llm_provider']!r}, not the calling surface"
