"""End-to-end through a REAL activated client.

THE TEST CLASS THIS SUITE WAS MISSING. Every dispatch test called
dispatch_sync directly, so they proved the component works and never that
activation wires it. The result was 391 passing tests over an agent that
refused every call in enforce and passed every call unguarded in monitor,
because the factories were handed no executor and the AttributeError was
filed as an internal invariant violation.

These tests go through `tidewall_otel.activate()` and a real provider client.
"""

import inspect
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

    # Residual instrumentors are module-level and survive deactivate() by
    # design -- that is the whole point of parking them. A test that leaves
    # one behind therefore makes the NEXT test's activate() park it too, and
    # the failure surfaces as an unrelated assertion three tests later.
    tidewall_otel._residual_instrumentors.clear()
    yield
    tidewall_otel.deactivate()

    # HARD reset. A test that restores an SDK attribute directly leaves the
    # manager holding an entry whose `installed` object is no longer anywhere,
    # so removal declines forever and the instrumentor is retained by design.
    # The next test's `activate()` then parks it, and the failure surfaces as
    # an unrelated assertion in a later test.
    tidewall_otel._residual_instrumentors.clear()
    tidewall_otel._instrumentor_instance = None


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


def test_a_failure_MID_ACTIVATION_leaves_no_live_partial_patch(monkeypatch):
    """Finding 2, at the activation seam rather than the manager's.

    `PatchManager.install_all` has always rolled back. The defect was that
    normal activation never called it -- it applied four independent
    `wrap_function_wrapper` calls, so a failure on the second left the first
    live on the SDK while the public lifecycle read `uninstalled`. A caller
    reading that state would believe the SDK was pristine while an orphaned
    Tidewall wrapper stayed installed for the life of the process.

    Testing the manager's rollback in isolation cannot catch this: the
    manager was correct and simply disconnected.
    """
    import inspect

    from openai.resources.chat.completions.completions import Completions

    from tidewall_otel._manager import PatchManager

    before = inspect.getattr_static(Completions, "create")

    real, calls = PatchManager.install, []

    def flaky(self, module, name, wrapper):
        calls.append(name)
        if len(calls) == 2:
            raise RuntimeError("synthetic second-patch failure")
        return real(self, module, name, wrapper)

    monkeypatch.setattr(PatchManager, "install", flaky)

    with pytest.raises(RuntimeError, match="synthetic second-patch failure"):
        tidewall_otel.activate()

    assert len(calls) == 2, "activation did not go through the manager at all"
    assert inspect.getattr_static(Completions, "create") is before, (
        "the first patch survived a failed activation"
    )
    assert tidewall_otel.state().lifecycle != "installed"
    assert tidewall_otel.is_active() is False


def test_an_escape_is_recorded_even_AFTER_an_unrelated_event(guard_says, provider):
    """Finding 2 of round 2. The escape bridge must not be suppressible.

    `and not state.events` added to the downgrade condition in `_prepare`
    survived the entire behavioural suite: every existing escape test recorded
    the escape as the FIRST event, so a guard reading "only downgrade while
    nothing has happened yet" was indistinguishable from a correct one. A prior
    skip -- a dry-run call, a lossy refusal, anything -- would then suppress
    every later downgrade while the surface stayed `covered`.
    """
    guard_says(CLEAN)
    client, _reached = provider

    tidewall_otel.activate()
    state = tidewall_otel.state()

    # An unrelated event FIRST, recorded through the public seam.
    state.record_skip("Messages.create", reason="unrelated", detail=None)
    assert state.events, "the precondition did not take"

    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert state.surfaces["Completions.create"] == "unverified", (
        "a prior event suppressed the escape downgrade"
    )
    assert any(e.reason == "client_escapes" for e in state.events)


def test_deactivation_that_could_NOT_remove_reports_residual_not_removed(monkeypatch):
    """Finding 1 of round 2. `removed` must mean removed.

    When another agent wraps a boundary after us, `PatchManager.remove()`
    correctly declines to write -- deleting their wrapper to reinstate ours
    would corrupt the stack. That leaves a Tidewall wrapper live on the SDK
    underneath theirs, and the instrumentor records it as a residual.

    Publishing an unconditional `lifecycle="removed"` threw that evidence away
    and told an operator the SDK was pristine while our code still ran on every
    call. This is the reporting-layer twin of claiming enforcement while
    unguarded, and the round-1 fixes did not touch it.
    """
    import wrapt
    from openai.resources.chat.completions.completions import Completions

    # This test deliberately creates a wrapper that removal REFUSES to touch --
    # that is the whole point -- so it must put the class back itself. Leaving
    # it means every later test in the session runs against a doubly-wrapped
    # SDK, which is exactly how three unrelated manifest tests failed the first
    # time this was written.
    pristine = inspect.getattr_static(Completions, "create")
    try:
        tidewall_otel.activate()

        # SOMEONE ELSE wraps the method after us; the last writer wins.
        wrapt.wrap_function_wrapper(
            "openai.resources.chat.completions.completions", "Completions.create",
            lambda wrapped, instance, args, kwargs: wrapped(*args, **kwargs))
        foreign = inspect.getattr_static(Completions, "create")

        tidewall_otel.deactivate()

        state = tidewall_otel.state()
        assert state.lifecycle == "residual", (
            f"deactivation reported {state.lifecycle!r} while a wrapper stayed installed"
        )
        assert any(e.reason == "not_removed" for e in state.events), state.events
        assert inspect.getattr_static(Completions, "create") is foreign
    finally:
        Completions.create = pristine

    assert inspect.getattr_static(Completions, "create") is pristine

    # And the honest case still reports removed.
    tidewall_otel.activate()
    tidewall_otel.deactivate()
    assert tidewall_otel.state().lifecycle == "removed"


def test_DRY_RUN_reaches_the_provider_and_NEVER_contacts_the_guard(
        monkeypatch, guard_says, provider):
    """Round 10's most serious finding: there was no activated dry-run test.

    Changing `config.mode` from `dry-run` to `monitor` immediately before
    dispatch -- which makes an activated dry-run call contact the guard, in
    direct violation of the documented mode contract -- left 139 tests green
    across end-to-end, activation, mode-policy, both dispatch suites and state.

    The monitor path had such a test; the mode whose entire promise is "no
    guard call" did not. A mode contract nothing exercises is a comment.
    """
    monkeypatch.setenv("TIDEWALL_MODE", "dry-run")
    asked = guard_says(CLEAN)
    client, reached = provider

    tidewall_otel.activate()
    result = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert asked == [], (
        f"dry-run contacted the guard {len(asked)} time(s); its whole "
        f"contract is that it does not")
    assert len(reached) == 1, "dry-run did not reach the provider"
    assert result.choices[0].message.content == "ok"

    state = tidewall_otel.state()
    assert state.mode == "dry-run"
    assert state.is_active() is False, (
        "dry-run must not claim to be enforcing -- it never asks the guard")


def test_deactivate_RETRIES_through_the_public_api_after_a_conflict_clears():
    """The retry must be reachable by the only API a caller has.

    An earlier fix made `_uninstrument()` keep its manager when entries could
    not be removed -- necessary, and not sufficient. Two barriers still stood
    in front of the public path: `deactivate()` discarded
    `_instrumentor_instance` regardless, and `BaseInstrumentor.uninstrument()`
    gates on a flag the first deactivation clears, so a second call did
    nothing at all.

    The regression test for that fix called the PRIVATE `_uninstrument()`
    twice and so walked past both barriers. It proved the manager retained
    the entry and nothing about whether anyone could ever use it -- the same
    defect as testing dispatch directly and never testing activation.
    """
    import inspect

    import wrapt
    from openai.resources.chat.completions.completions import Completions

    pristine = inspect.getattr_static(Completions, "create")
    try:
        tidewall_otel.activate()

        # Another agent wraps on top, so removal must decline.
        wrapt.wrap_function_wrapper(
            "openai.resources.chat.completions.completions", "Completions.create",
            lambda wrapped, instance, args, kwargs: "foreign")
        foreign = inspect.getattr_static(Completions, "create")

        tidewall_otel.deactivate()

        assert tidewall_otel._instrumentor_instance is not None, (
            "the instrumentor was discarded, so no retry is possible")
        assert inspect.getattr_static(Completions, "create") is foreign, (
            "deactivation deleted a wrapper installed after ours")
        assert tidewall_otel.state().lifecycle == "residual"

        # The conflicting wrapper goes; the public API must finish the job.
        Completions.create = foreign.__wrapped__
        tidewall_otel.deactivate()

        assert inspect.getattr_static(Completions, "create") is pristine, (
            "the retry did not restore the original attribute")
        assert tidewall_otel._instrumentor_instance is None, (
            "a fully discharged instrumentor was retained")
        assert tidewall_otel.state().lifecycle == "removed"
    finally:
        Completions.create = pristine


def test_REACTIVATION_does_not_orphan_a_retained_removal_journal():
    """The full public sequence the previous fixes still failed.

    activate -> foreign wrapper -> deactivate (residual, journal retained)
    -> ACTIVATE AGAIN -> conflict clears -> deactivate.

    `activate()` blocked only on lifecycle `installed`, so a re-activation
    while `residual` replaced `_instrumentor_instance` outright. That
    destroyed the only route to the retained journal and its
    `pre_install_identity`: when the foreign wrapper was later removed, our
    stale wrapper became live again with nothing able to remove it -- and the
    new activation reported `installed`, hiding it.

    Parking rather than blocking is deliberate. Refusing to activate while a
    residual exists would let one stuck foreign wrapper leave the process
    unguarded for the rest of its life, trading a stale layer for no layer.
    """
    import inspect

    import wrapt
    from openai.resources.chat.completions.completions import Completions

    pristine = inspect.getattr_static(Completions, "create")
    try:
        tidewall_otel.activate()
        wrapt.wrap_function_wrapper(
            "openai.resources.chat.completions.completions", "Completions.create",
            lambda wrapped, instance, args, kwargs: "foreign")
        foreign = inspect.getattr_static(Completions, "create")

        tidewall_otel.deactivate()
        assert tidewall_otel.state().lifecycle == "residual"

        tidewall_otel.activate()
        assert tidewall_otel._residual_instrumentors, (
            "the retained journal was orphaned by re-activation")

        # The conflict clears; deactivation must discharge BOTH the live
        # instrumentor and the parked one.
        Completions.create = foreign.__wrapped__
        tidewall_otel.deactivate()

        assert inspect.getattr_static(Completions, "create") is pristine, (
            "a stale wrapper survived every deactivation")
        assert tidewall_otel._residual_instrumentors == [], (
            "a discharged residual was not released")
    finally:
        Completions.create = pristine
        tidewall_otel._residual_instrumentors.clear()


def test_activation_still_RETRIES_a_residual_before_parking_it():
    """If the conflict has already cleared, re-activation should discharge
    the residual outright rather than accumulate it."""
    import inspect

    import wrapt
    from openai.resources.chat.completions.completions import Completions

    pristine = inspect.getattr_static(Completions, "create")
    try:
        tidewall_otel.activate()
        wrapt.wrap_function_wrapper(
            "openai.resources.chat.completions.completions", "Completions.create",
            lambda wrapped, instance, args, kwargs: "foreign")
        foreign = inspect.getattr_static(Completions, "create")
        tidewall_otel.deactivate()

        Completions.create = foreign.__wrapped__      # clears BEFORE re-activating
        tidewall_otel.activate()

        assert tidewall_otel._residual_instrumentors == [], (
            "a residual whose conflict had cleared was parked anyway")
    finally:
        tidewall_otel.deactivate()
        Completions.create = pristine
        tidewall_otel._residual_instrumentors.clear()
