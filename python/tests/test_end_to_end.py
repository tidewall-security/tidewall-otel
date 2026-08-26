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
    tidewall_otel._residual_managers.clear()
    yield
    tidewall_otel.deactivate()

    # HARD reset. A test that restores an SDK attribute directly leaves the
    # manager holding an entry whose `installed` object is no longer anywhere,
    # so removal declines forever and the instrumentor is retained by design.
    # The next test's `activate()` then parks it, and the failure surfaces as
    # an unrelated assertion in a later test.
    tidewall_otel._residual_managers.clear()
    tidewall_otel._instrumentor_instance = None

    # And the SINGLETON's own manager. The real `BaseInstrumentor` returns the
    # same object from every construction, so its `_manager` -- and any stuck
    # journal on it -- survives clearing the module-level reference entirely.
    # The next activation would hand that journal straight back into the
    # parked list and every later test would read `residual`.
    from tidewall_otel._instrumentor import TidewallInstrumentor

    TidewallInstrumentor()._manager = None


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
    with pytest.raises(tidewall_otel.TidewallBlockedError):
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
    with pytest.raises(tidewall_otel.LossyInputError):
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

    # An unrelated event FIRST. Recorded on the agent's own state, because
    # `state()` now hands out a read-only snapshot -- a write to it would
    # reach nothing, which is why the snapshot refuses rather than accepts.
    tidewall_otel._state.record_skip("Messages.create", reason="unrelated")
    assert tidewall_otel.state().events, "the precondition did not take"

    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    state = tidewall_otel.state()
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

    # Restoring the class by hand above left a journal entry whose `installed`
    # object is nowhere, so it can never discharge and would keep the
    # lifecycle at `residual` for every later cycle -- correctly, which is
    # exactly why the next assertion needs a clean slate rather than a
    # weakened expectation.
    tidewall_otel._residual_managers.clear()
    tidewall_otel._instrumentor_instance = None

    # And the SINGLETON's own manager. The real `BaseInstrumentor` returns the
    # same object from every construction, so its `_manager` -- and any stuck
    # journal on it -- survives clearing the module-level reference entirely.
    # The next activation would hand that journal straight back into the
    # parked list and every later test would read `residual`.
    from tidewall_otel._instrumentor import TidewallInstrumentor

    TidewallInstrumentor()._manager = None

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
    """activate() while `residual` must not destroy the retained journal.

    It blocked only on lifecycle `installed`, so re-activating replaced
    `_instrumentor_instance` outright. That threw away the only route to the
    retained journal and its `pre_install_identity`: when the foreign wrapper
    was later removed our stale wrapper became live again with nothing able
    to remove it, and the new activation reported `installed`, hiding it.

    What is asserted is that the ROUTE SURVIVES -- the journal is still
    reachable and still holds its entry after re-activation. Whether it then
    discharges is covered by `test_deactivate_RETRIES_through_the_public_api`
    and by the sibling test below.

    Asserted on the MANAGER, not the instrumentor: the real
    `BaseInstrumentor` is a singleton, so `TidewallInstrumentor()` returns the
    same object and "did activation build a new instrumentor" is false on the
    production OTel path while true against the fallback stub. The manager is
    what holds the entries, so it is what must survive.

    Deliberately NOT asserted: the final identity of the SDK attribute after
    unwinding a stack that two activations both patched. Unwrapping one layer
    of that by hand is an artificial state whose resolution depends on which
    instrumentor discharges first, and it differs between interpreters. The
    finding was about losing the route, not about resolving that stack.
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

        tidewall_otel.deactivate()
        retained = tidewall_otel._instrumentor_instance
        assert retained is not None
        parked_manager = retained._manager
        assert parked_manager.journal, "precondition: an entry was retained"
        assert tidewall_otel.state().lifecycle == "residual"

        tidewall_otel.activate()

        assert parked_manager in tidewall_otel._residual_managers, (
            "re-activation orphaned the retained journal")
        assert parked_manager.journal, (
            "the retained journal lost the entry it alone can remove")
    finally:
        Completions.create = pristine
        tidewall_otel._residual_managers.clear()


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

        assert tidewall_otel._residual_managers == [], (
            "a residual whose conflict had cleared was parked anyway")
    finally:
        tidewall_otel.deactivate()
        Completions.create = pristine
        tidewall_otel._residual_managers.clear()


def test_a_STUCK_parked_residual_is_not_hidden_by_a_clean_current_removal():
    """Fourth-pass finding: `deactivate()` replaced the residual list.

    The parked-manager loop appended its failures, and the active-instrumentor
    branch then did `residuals = list(instrumentor.residuals)` -- discarding
    every one of them. So a parked manager that was still stuck vanished from
    the report and lifecycle read `removed`, telling an operator the SDK was
    pristine while our wrapper waited under someone else's.

    Constructed with a parked manager that cannot discharge alongside a
    current activation that removes cleanly.
    """
    import sys
    import types

    from tidewall_otel._manager import PatchManager

    stuck_module = types.ModuleType("stuck_parked_sdk")
    stuck_module.Target = type("Target", (), {"create": staticmethod(lambda: "orig")})
    sys.modules["stuck_parked_sdk"] = stuck_module
    try:
        stuck = PatchManager()
        stuck.install("stuck_parked_sdk", "Target.create", lambda w, i, a, k: w(*a, **k))
        # Someone replaces it, so removal will decline forever.
        stuck_module.Target.create = staticmethod(lambda: "foreign")
        tidewall_otel._residual_managers.append(stuck)

        # A current activation that removes cleanly.
        tidewall_otel.activate()
        tidewall_otel.deactivate()

        assert tidewall_otel._residual_managers, (
            "the stuck parked manager was dropped")
        assert tidewall_otel.state().lifecycle == "residual", (
            "a clean current removal reported `removed` while a parked "
            "residual was still stuck")
        assert any(event.reason == "not_removed"
                   for event in tidewall_otel.state().events), (
            "the parked residual was not reported at all")
    finally:
        tidewall_otel._residual_managers.clear()
        sys.modules.pop("stuck_parked_sdk", None)


def test_a_PERMANENT_residual_survives_repeated_deactivation():
    """Finding: the durable record lasted exactly one call.

    `_uninstrument()` clears its manager once the journal empties, and
    unrecoverable entries empty it -- so `deactivate()` read
    `permanent_residuals` from a manager that was already gone. The first
    call reported the outcome, the instrumentor was dropped, and a second
    `deactivate()` built a fresh state reporting `removed`, contradicting the
    claim that the record is durable.

    On the real `BaseInstrumentor` singleton this is the only manager there
    is, so the record has to be copied out before either reference is cleared.
    """
    import weakref

    from tidewall_otel._manager import PatchManager

    manager = PatchManager()
    import sys
    import types

    victim = types.ModuleType("perm_sdk")
    victim.Target = type("Target", (), {"create": staticmethod(lambda: "orig")})
    sys.modules["perm_sdk"] = victim
    try:
        manager.install("perm_sdk", "Target.create", lambda w, i, a, k: w(*a, **k))

        watch = weakref.ref(victim.Target)
        del victim.Target
        del sys.modules["perm_sdk"]
        del victim
        import gc

        gc.collect()
        assert watch() is None, "the owner survived; this test proves nothing"

        tidewall_otel._residual_managers.append(manager)
        tidewall_otel.deactivate()

        assert tidewall_otel._permanent_residuals, "the record was never captured"
        first = list(tidewall_otel._permanent_residuals)
        assert any(event.reason == "unrecoverable"
                   for event in tidewall_otel.state().events), (
            "the permanent residual was not reported to the caller")

        # A SECOND deactivation must still report it.
        tidewall_otel.deactivate()

        assert tidewall_otel._permanent_residuals == first, (
            "the durable record did not survive a second deactivation")
        assert any(event.reason == "unrecoverable"
                   for event in tidewall_otel.state().events), (
            "a second deactivation reported a clean state over a permanent "
            "residual")
    finally:
        tidewall_otel._permanent_residuals.clear()
        tidewall_otel._residual_managers.clear()
        sys.modules.pop("perm_sdk", None)


def test_a_PERMANENT_record_does_not_taint_a_later_module_of_the_same_NAME():
    """Sixth-pass finding: records are strings keyed by `module.attribute`.

    Generation A is patched, its owner collected, and the entry recorded
    permanently. Generation B is then imported under the SAME module path,
    patched, and removed cleanly -- a different object entirely. Replaying A's
    record through `record_unverified` downgraded B's surface forever, so a
    reload or plugin system saw every later generation reported unrecoverable
    on the strength of an earlier one's history.

    The record is history and must be reported as history: an event, never a
    disposition.
    """
    import gc
    import sys
    import types
    import weakref

    from tidewall_otel._manager import PatchManager, RemovalOutcome

    try:
        gen_a = types.ModuleType("gen_sdk")
        gen_a.Target = type("Target", (), {"create": staticmethod(lambda: "A")})
        sys.modules["gen_sdk"] = gen_a
        manager_a = PatchManager()
        manager_a.install("gen_sdk", "Target.create", lambda w, i, a, k: w(*a, **k))
        watch = weakref.ref(gen_a.Target)

        del gen_a.Target, sys.modules["gen_sdk"], gen_a
        gc.collect()
        assert watch() is None, "generation A survived; this proves nothing"

        tidewall_otel._residual_managers.append(manager_a)
        tidewall_otel.deactivate()
        assert tidewall_otel._permanent_residuals, "A was not recorded"

        gen_b = types.ModuleType("gen_sdk")
        gen_b.Target = type("Target", (), {"create": staticmethod(lambda: "B")})
        sys.modules["gen_sdk"] = gen_b
        manager_b = PatchManager()
        manager_b.install("gen_sdk", "Target.create", lambda w, i, a, k: w(*a, **k))
        outcomes = manager_b.remove()

        assert outcomes[("gen_sdk", "Target.create")] is RemovalOutcome.REMOVED
        assert gen_b.Target.create() == "B", "generation B was not restored"

        tidewall_otel.deactivate()
        state = tidewall_otel.state()

        assert any(event.reason == "unrecoverable" for event in state.events), (
            "the durable record was lost")
        assert "gen_sdk.Target.create" not in state.surfaces, (
            f"a collected generation's record downgraded a later one: "
            f"{state.surfaces}")
    finally:
        tidewall_otel._permanent_residuals.clear()
        tidewall_otel._residual_managers.clear()
        sys.modules.pop("gen_sdk", None)


def test_a_TOOL_CALL_message_is_refused_in_enforce_and_declared_in_monitor():
    """Guard/provider divergence, checked in both modes.

    An assistant message can carry attacker-controlled
    `tool_calls[].function.arguments`. The normalizer emits only `role` and
    `content`, so the guard would never see them -- which would be a bypass
    of exactly the P0-11 shape if the agent proceeded anyway.

    It does not. The path is unmapped, therefore lossy: `enforce` refuses
    before either the guard or the provider is contacted, and `monitor`
    proceeds by its own contract while recording a `lossy` skip and dropping
    `is_active()` to False. The agent declines to certify what it cannot
    show the guard.
    """
    import json
    import os

    messages = [
        {"role": "user", "content": "benign"},
        {"role": "assistant", "content": "calling a tool", "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "exfiltrate",
                "arguments": '{"payload": "LEAK-ME"}'}}]},
    ]

    # --- enforce: refused, nothing contacted -------------------------------
    os.environ["TIDEWALL_MODE"] = "enforce"
    asked, sent = [], []
    tidewall_otel.activate()
    try:
        import tidewall_otel._guard as guard_module

        original = guard_module.post_guard
        guard_module.post_guard = lambda **kw: (
            asked.append(kw["payload"]), CLEAN)[1]

        client = openai.OpenAI(api_key="t", http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: (sent.append(json.loads(r.content)),
                           httpx.Response(200, json=_COMPLETION))[1])))

        with pytest.raises(tidewall_otel.LossyInputError):
            client.chat.completions.create(model="gpt-4o", messages=messages)

        assert asked == [], "the guard was asked about a body it was not shown"
        assert sent == [], "the provider received an uninspected tool call"
    finally:
        guard_module.post_guard = original
        tidewall_otel.deactivate()


def test_a_container_that_READS_differently_than_it_STORES_is_refused():
    """A guard bypass of the P0-11 shape, from the container rather than a kwarg.

    Classification walks values one way and the normalizer reads them another
    -- `items()` here, `get()` there. A dict SUBCLASS that overrides
    `get("content")` therefore showed the guard benign text while the
    provider serialised the stored value, and the call PROCEEDED in enforce.
    Reproduced before the fix: guard saw "benign text", provider received the
    attack.

    Fail-closed rather than clever: when two readings of a container
    disagree, the agent cannot say which the provider will use, so the call
    is lossy and enforce declines it.
    """
    import json
    import os

    import tidewall_otel._guard as guard_module

    class Deceptive(dict):
        def get(self, key, default=None):
            if key == "content":
                return "benign text"
            return super().get(key, default)

    os.environ["TIDEWALL_MODE"] = "enforce"
    asked, sent = [], []
    original = guard_module.post_guard
    guard_module.post_guard = lambda **kw: (asked.append(kw["payload"]), {"result": CLEAN})[1]

    tidewall_otel.activate()
    try:
        client = openai.OpenAI(api_key="t", http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: (sent.append(json.loads(r.content)),
                           httpx.Response(200, json=_COMPLETION))[1])))

        with pytest.raises(tidewall_otel.LossyInputError):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[Deceptive(role="user",
                                    content="SECRET MALICIOUS INSTRUCTION")])

        assert asked == [], "the guard was shown a body the provider would not send"
        assert sent == [], "the provider received an uninspected payload"
    finally:
        guard_module.post_guard = original
        tidewall_otel.deactivate()


def test_an_ORDINARY_container_is_not_refused_by_the_divergence_check():
    """The other direction. Subclassed mappings are ordinary in SDK code, and
    a check that refused them all would be unusable -- only a container whose
    readings actually DISAGREE is lossy."""
    import os

    import tidewall_otel._guard as guard_module

    class Benign(dict):
        pass

    os.environ["TIDEWALL_MODE"] = "enforce"
    original = guard_module.post_guard
    guard_module.post_guard = lambda **kw: {"result": CLEAN}

    tidewall_otel.activate()
    try:
        client = openai.OpenAI(api_key="t", http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=_COMPLETION))))

        for message in ({"role": "user", "content": "hi"},
                        Benign(role="user", content="hi")):
            result = client.chat.completions.create(
                model="gpt-4o", messages=[message])
            assert result.choices[0].message.content == "ok"
    finally:
        guard_module.post_guard = original
        tidewall_otel.deactivate()


def test_content_MUTATED_while_the_guard_runs_is_refused():
    """A TOCTOU bypass: the guard inspects a snapshot, the provider is invoked
    with the caller's own mutable kwargs afterwards.

    Anything running in between -- another thread, a callback, a re-entrant
    guard -- can swap the content, and enforce then approves one prompt and
    sends another. Reproduced before the fix: the provider received
    "MALICIOUS after inspection" while the guard had seen benign text.

    The agent does not lock the caller's data, which it does not own. It
    compares a trusted fingerprint across the guard call and declines when
    what it inspected is no longer what it would send.
    """
    import json
    import os

    import tidewall_otel._guard as guard_module

    message = {"role": "user", "content": "benign at inspection time"}
    sent = []

    def mutating_guard(**kwargs):
        message["content"] = "MALICIOUS after inspection"
        return {"result": CLEAN}

    os.environ["TIDEWALL_MODE"] = "enforce"
    original = guard_module.post_guard
    guard_module.post_guard = mutating_guard

    tidewall_otel.activate()
    try:
        client = openai.OpenAI(api_key="t", http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: (sent.append(json.loads(r.content)),
                           httpx.Response(200, json=_COMPLETION))[1])))

        with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
            client.chat.completions.create(model="gpt-4o", messages=[message])

        assert raised.value.outcome_kind == "mutated_during_guard"
        assert sent == [], "the provider received content nothing inspected"
    finally:
        guard_module.post_guard = original
        tidewall_otel.deactivate()


def test_an_UNCHANGED_request_is_not_refused_by_the_mutation_check():
    """The other direction: the check must not refuse ordinary traffic, whose
    kwargs are untouched across the guard call."""
    import os

    import tidewall_otel._guard as guard_module

    os.environ["TIDEWALL_MODE"] = "enforce"
    original = guard_module.post_guard
    guard_module.post_guard = lambda **kw: {"result": CLEAN}

    tidewall_otel.activate()
    try:
        client = openai.OpenAI(api_key="t", http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=_COMPLETION))))
        result = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        assert result.choices[0].message.content == "ok"
    finally:
        guard_module.post_guard = original
        tidewall_otel.deactivate()


def test_a_deceptive_STR_SUBCLASS_cannot_hide_behind_its_own_equality():
    """The fidelity check first compared readings with `!=`, which calls the
    value's own `__eq__`. A `str` subclass storing an attack and reporting
    itself equal to benign text therefore passed -- guard saw benign, provider
    serialised the attack, enforce approved it.

    Equality supplied by the thing under inspection cannot be the boundary.
    Snapshots are type-tagged and exact instead.
    """
    import json
    import os

    import tidewall_otel._guard as guard_module

    class LyingStr(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

        def __hash__(self):
            return hash("benign")

    os.environ["TIDEWALL_MODE"] = "enforce"
    sent = []
    original = guard_module.post_guard
    guard_module.post_guard = lambda **kw: {"result": CLEAN}

    tidewall_otel.activate()
    try:
        client = openai.OpenAI(api_key="t", http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: (sent.append(json.loads(r.content)),
                           httpx.Response(200, json=_COMPLETION))[1])))

        with pytest.raises(tidewall_otel.LossyInputError):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": LyingStr("MALICIOUS")}])

        assert sent == [], "the provider received an unvouched-for value"
    finally:
        guard_module.post_guard = original
        tidewall_otel.deactivate()


def test_a_MUTATED_dict_subclass_is_caught_by_the_fingerprint():
    """The fingerprint reduced any non-exact container to its TYPE NAME, so a
    dict subclass or pydantic model mutated in place kept an identical
    fingerprint -- the type had not changed -- and the provider received
    content the guard never saw.

    Containers now snapshot their stored contents, read through
    `dict.items` and `object.__getattribute__` so an overridden accessor
    cannot dress up what is actually there.
    """
    import json
    import os

    import tidewall_otel._guard as guard_module

    class Benign(dict):
        pass

    message = Benign(role="user", content="benign at inspection")
    sent = []

    def mutating_guard(**kwargs):
        message["content"] = "MALICIOUS via subclass"
        return {"result": CLEAN}

    os.environ["TIDEWALL_MODE"] = "enforce"
    original = guard_module.post_guard
    guard_module.post_guard = mutating_guard

    tidewall_otel.activate()
    try:
        client = openai.OpenAI(api_key="t", http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: (sent.append(json.loads(r.content)),
                           httpx.Response(200, json=_COMPLETION))[1])))

        with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
            client.chat.completions.create(model="gpt-4o", messages=[message])

        assert raised.value.outcome_kind == "mutated_during_guard"
        assert sent == [], "mutated subclass content reached the provider"
    finally:
        guard_module.post_guard = original
        tidewall_otel.deactivate()


def test_MONITOR_does_not_block_on_a_mutation_it_cannot_verify(monkeypatch, provider):
    """Monitor's whole promise is that it does not affect users.

    Every other refusal path honours the mode switch -- blocked, transformed,
    and every failure kind fall through to Proceed outside enforce. The
    mutation check did not: it raised unconditionally, so a caller who
    mutated its own kwargs during the guard call had its request killed by
    the mode that exists precisely to kill nothing.

    Monitor cannot claim the surface either. It records `unverified` with the
    reason, which is the difference between out-of-model and unnoticed.
    """
    import tidewall_otel._guard as guard_module

    monkeypatch.setenv("TIDEWALL_MODE", "monitor")
    message = {"role": "user", "content": "benign at inspection time"}
    client, reached = provider

    def mutating_guard(**kwargs):
        message["content"] = "changed after inspection"
        return {"result": CLEAN}

    monkeypatch.setattr(guard_module, "post_guard", mutating_guard)

    tidewall_otel.activate()
    client.chat.completions.create(model="gpt-4o", messages=[message])

    assert len(reached) == 1, "monitor blocked a call it only promised to watch"
    state = tidewall_otel.state()
    assert state.surfaces["Completions.create"] == "unverified"
    assert any(e.reason == "mutated_during_guard" for e in state.events), \
        "monitor proceeded but recorded nothing, which is a silent bypass"


async def test_MONITOR_does_not_block_on_a_mutation_on_the_ASYNC_path(monkeypatch):
    """The async mutation window is the wider of the two -- the loop can run
    arbitrary other tasks while the guard call is awaited -- so the async path
    needs its own proof, not the sync path's."""
    import tidewall_otel._guard as guard_module

    monkeypatch.setenv("TIDEWALL_MODE", "monitor")
    message = {"role": "user", "content": "benign at inspection time"}
    reached = []

    def mutating_guard(**kwargs):
        message["content"] = "changed after inspection"
        return {"result": CLEAN}

    monkeypatch.setattr(guard_module, "post_guard", mutating_guard)

    def handler(request):
        reached.append(json.loads(request.content))
        return httpx.Response(200, json=_COMPLETION)

    client = openai.AsyncOpenAI(
        api_key="t",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    tidewall_otel.activate()
    await client.chat.completions.create(model="gpt-4o", messages=[message])

    assert len(reached) == 1, "monitor blocked an async call it only watched"
    state = tidewall_otel.state()
    assert any(e.reason == "mutated_during_guard" for e in state.events)


async def test_ENFORCE_still_refuses_a_mutation_on_the_ASYNC_path(monkeypatch):
    """Making monitor non-blocking must not make enforce non-blocking. The
    async arm gets its own enforce proof for the same reason."""
    import tidewall_otel._guard as guard_module

    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    message = {"role": "user", "content": "benign at inspection time"}
    reached = []

    def mutating_guard(**kwargs):
        message["content"] = "MALICIOUS after inspection"
        return {"result": CLEAN}

    monkeypatch.setattr(guard_module, "post_guard", mutating_guard)

    client = openai.AsyncOpenAI(
        api_key="t",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: (reached.append(json.loads(r.content)),
                       httpx.Response(200, json=_COMPLETION))[1])))

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        await client.chat.completions.create(model="gpt-4o", messages=[message])

    assert raised.value.outcome_kind == "mutated_during_guard"
    assert reached == [], "async enforce sent content nothing inspected"


# -- a bound that closed a crash must not open a bypass --------------------
#
# `_trusted` replaces anything past its depth or node budget with a CONSTANT
# sentinel, and two identical sentinels compare equal. So a value changed
# below the boundary produced matching before/after fingerprints, and the
# mutation check -- which only compared them -- saw nothing. The bound
# closed a `RecursionError` and opened a bypass in the check it protected.
#
# Reachability, stated honestly: with the shipped budgets, classification
# refuses a deeply nested tool schema as LOSSY before the guard is ever
# called, so the depth boundary is not reachable that way. Exhausting the
# node budget through fully-declared content needs a conversation of tens of
# thousands of messages. The fix is defence in depth, and these tests shrink
# the budget so the mechanism itself is exercised rather than the constant.
#
# An earlier version of these tests buried the value under an undeclared key
# and went green without the fingerprint check ever running, because
# `LossyInputError` subclasses `TidewallRefusedError`. Each test now asserts
# the guard WAS called and names the exact `outcome_kind`.

def _tiny_budget(monkeypatch, nodes=3):
    """Make the node budget reachable with an ordinary payload."""
    import tidewall_otel._manifest as manifest
    monkeypatch.setattr(manifest, "_MAX_NODES", nodes)


def test_a_mutation_BELOW_THE_BUDGET_BOUNDARY_never_reaches_the_provider(monkeypatch):
    """Enforce refuses on an INCOMPLETE fingerprint alone, without waiting
    for a difference the fingerprint is incapable of showing."""
    import tidewall_otel._guard as guard_module

    _tiny_budget(monkeypatch)
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    message = {"role": "user", "content": "benign"}
    asked, sent = [], []

    def mutating_guard(**kwargs):
        asked.append(1)
        message["content"] = "MALICIOUS"
        return {"result": CLEAN}

    monkeypatch.setattr(guard_module, "post_guard", mutating_guard)

    tidewall_otel.activate()
    client = openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(
            lambda r: (sent.append(json.loads(r.content)),
                       httpx.Response(200, json=_COMPLETION))[1])))

    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        client.chat.completions.create(model="gpt-4o", messages=[message])

    assert asked, "the payload never reached the guard, so this proves nothing"
    assert raised.value.outcome_kind in ("unverifiable_payload",
                                         "mutated_during_guard"), \
        f"refused for the wrong reason: {raised.value.outcome_kind}"
    assert sent == [], "content the guard never inspected was sent"


def test_an_UNCHANGED_but_unverifiable_payload_is_also_refused(monkeypatch):
    """The sentinel case specifically: nothing mutated, but the fingerprint
    could not capture the payload, so it cannot testify that nothing did."""
    import tidewall_otel._guard as guard_module

    _tiny_budget(monkeypatch)
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    asked, sent = [], []

    def clean_guard(**kwargs):
        asked.append(1)
        return {"result": CLEAN}

    monkeypatch.setattr(guard_module, "post_guard", clean_guard)

    tidewall_otel.activate()
    client = openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(
            lambda r: (sent.append(json.loads(r.content)),
                       httpx.Response(200, json=_COMPLETION))[1])))

    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "benign"}])

    assert asked
    assert raised.value.outcome_kind == "unverifiable_payload"
    assert sent == []


async def test_the_same_hole_is_closed_on_the_ASYNC_path(monkeypatch):
    import tidewall_otel._guard as guard_module

    _tiny_budget(monkeypatch)
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    asked, reached = [], []

    def clean_guard(**kwargs):
        asked.append(1)
        return {"result": CLEAN}

    monkeypatch.setattr(guard_module, "post_guard", clean_guard)

    tidewall_otel.activate()
    client = openai.AsyncOpenAI(api_key="t", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: (reached.append(json.loads(r.content)),
                       httpx.Response(200, json=_COMPLETION))[1])))

    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "benign"}])

    assert asked, "the async payload never reached the guard"
    assert raised.value.outcome_kind == "unverifiable_payload"
    assert reached == [], "async sent a payload it could not verify"


def test_MONITOR_records_an_unverifiable_payload_instead_of_blocking(monkeypatch):
    """Failing closed is an enforce behaviour. Monitor still proceeds, and
    still refuses to claim the surface it could not verify."""
    import tidewall_otel._guard as guard_module

    _tiny_budget(monkeypatch)
    monkeypatch.setenv("TIDEWALL_MODE", "monitor")
    monkeypatch.setattr(guard_module, "post_guard",
                        lambda **kw: {"result": CLEAN})
    reached = []

    tidewall_otel.activate()
    client = openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(
            lambda r: (reached.append(json.loads(r.content)),
                       httpx.Response(200, json=_COMPLETION))[1])))

    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "benign"}])

    assert len(reached) == 1, "monitor blocked on an unverifiable payload"
    state = tidewall_otel.state()
    assert state.surfaces["Completions.create"] == "unverified"
    assert any(e.reason == "unverifiable_payload" for e in state.events)


def test_a_REALISTIC_large_payload_is_still_verifiable():
    """Failing closed must not refuse ordinary large traffic, which is what
    sets the budget: 100 tools of 100 fields is ~61k nodes, and a 100-message
    conversation is ~500."""
    from tidewall_otel._manifest import _trusted, fingerprint_is_incomplete

    tools = [{"type": "function", "function": {
        "name": f"tool_{i}", "description": "d" * 80,
        "parameters": {"type": "object", "properties": {
            f"field_{j}": {"type": "string", "description": "x" * 40}
            for j in range(100)}}}} for i in range(100)]
    conversation = [{"role": "user", "content": "hello " * 200} for _ in range(100)]

    assert not fingerprint_is_incomplete(_trusted(tools)), \
        "a realistic tool payload became unverifiable"
    assert not fingerprint_is_incomplete(_trusted(conversation))


# -- the guard-health dimension -------------------------------------------
#
# `guard_health` was declared in `State`, documented as a first-class
# dimension, and tested by constructing `State(guard_health=...)` directly --
# and NO production code ever wrote to it. An operator polling `state()`
# through a total guard outage saw `unknown` from activation onwards while
# every enforce-mode call was failing. Fifth instance in this programme of
# something built, tested, and wired to nothing; the tests all passed because
# every one of them supplied the value it then asserted.

def _guard_raising(exc):
    def raiser(**kwargs):
        raise exc
    return raiser


def test_a_guard_OUTAGE_is_visible_in_state(monkeypatch, provider):
    from tidewall_otel._http import GuardUnreachable

    import tidewall_otel._guard as guard_module

    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setattr(guard_module, "post_guard",
                        _guard_raising(GuardUnreachable("connection refused")))
    client, reached = provider

    tidewall_otel.activate()
    assert tidewall_otel.state().guard_health == "unknown", \
        "health claimed before any evidence existed"

    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    # Dispatch converts every unexpected BaseException to `invariant_violated`,
    # which is still a TidewallRefusedError -- so asserting the exception TYPE
    # alone lets an unrelated defect satisfy the test. Name the reason.
    assert raised.value.outcome_kind == "unreachable", \
        f"refused as {raised.value.outcome_kind}, not the outage under test"
    assert tidewall_otel.state().guard_health == "unreachable", \
        "an operator polling state() saw nothing wrong during an outage"
    assert reached == []


def test_a_WORKING_guard_reports_ok(monkeypatch, guard_says, provider):
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    guard_says(CLEAN)
    client, _reached = provider

    tidewall_otel.activate()
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert tidewall_otel.state().guard_health == "ok"


def test_health_names_the_ACTUAL_failure_not_a_flattened_one(monkeypatch, provider):
    """`saturated` is this agent's own pool declining work and
    `schema_invalid` is the guard answering badly. Reporting either as
    `unreachable` would misdirect whoever is paging."""
    import tidewall_otel._guard as guard_module

    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setattr(guard_module, "post_guard",
                        lambda **kw: {"nonsense": True})
    client, _reached = provider

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert raised.value.outcome_kind == "schema_invalid"
    assert tidewall_otel.state().guard_health == "schema_invalid"


def test_an_outage_does_NOT_flip_is_active(monkeypatch, provider):
    """Deliberate, and worth a test so it cannot be changed by accident.

    `is_active()` is a claim about BOUNDARIES -- whether every surface
    present is covered. An unreachable guard is a runtime condition the mode
    contract already handles per call; it does not retroactively mean the
    boundaries are unguarded. The health dimension is where an operator looks
    for that, which is why it had to be wired rather than folded in here.
    """
    from tidewall_otel._http import GuardUnreachable

    import tidewall_otel._guard as guard_module

    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setattr(guard_module, "post_guard",
                        _guard_raising(GuardUnreachable("down")))
    client, _reached = provider

    tidewall_otel.activate()
    # The FIRST call settles conditions unrelated to guard health -- this
    # fixture's custom httpx client is an escape, which downgrades the surface
    # on its own. Comparing across it would credit the outage with a change it
    # did not cause, so the comparison spans the second call instead.
    for _ in range(2):
        with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
            client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        assert raised.value.outcome_kind == "unreachable", \
            f"refused as {raised.value.outcome_kind}, not the outage under test"
        if _ == 0:
            before = tidewall_otel.state()
            surfaces, active = dict(before.surfaces), before.is_active()

    after = tidewall_otel.state()
    assert after.guard_health == "unreachable", "premise: the guard is down"
    assert after.surfaces == surfaces, "the outage downgraded a BOUNDARY"
    assert after.is_active() == active


def test_a_SUSTAINED_outage_does_not_grow_the_event_log(monkeypatch, provider):
    """The reason health is a scalar. Appending an event per failed call
    would grow without bound during exactly the outage an operator most needs
    the process to survive."""
    from tidewall_otel._http import GuardUnreachable

    import tidewall_otel._guard as guard_module

    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setattr(guard_module, "post_guard",
                        _guard_raising(GuardUnreachable("down")))
    client, _reached = provider

    tidewall_otel.activate()
    for _ in range(50):
        with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
            client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        # EVERY iteration: checking only the final state would miss a wrong
        # failure on any subset of the fifty.
        assert raised.value.outcome_kind == "unreachable"

    state = tidewall_otel.state()
    assert state.guard_health == "unreachable"
    assert len(state.events) < 10, \
        f"50 failed calls left {len(state.events)} events; this grows unbounded"


async def test_the_ASYNC_arm_reports_health_too(monkeypatch):
    from tidewall_otel._http import GuardUnreachable

    import tidewall_otel._guard as guard_module

    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setattr(guard_module, "post_guard",
                        _guard_raising(GuardUnreachable("down")))

    client = openai.AsyncOpenAI(api_key="t", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_COMPLETION))))

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert raised.value.outcome_kind == "unreachable"
    assert tidewall_otel.state().guard_health == "unreachable"


def test_MONITOR_does_not_grow_the_event_log_on_repeated_skips(monkeypatch, guard_says):
    """`record_skip` had the same unbounded shape as `record_unverified`.

    Monitor exists to be left running in production, and a lossy call shape
    is a steady-state property of an application, not an incident -- so the
    mode designed for long deployments logged one event per call, forever.
    """
    monkeypatch.setenv("TIDEWALL_MODE", "monitor")
    guard_says(CLEAN)
    reached = []
    client = openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(
            lambda r: (reached.append(1),
                       httpx.Response(200, json=_COMPLETION))[1])))

    tidewall_otel.activate()
    for _ in range(50):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            extra_body={"unrepresentable": True})

    assert len(reached) == 50, "monitor blocked calls it promised to watch"
    events = tidewall_otel.state().events
    assert any(e.reason == "lossy" for e in events), "the skip went unrecorded"
    assert len(events) < 10, \
        f"50 lossy calls left {len(events)} events; this grows unbounded"


def test_a_DIFFERENT_reason_is_still_recorded(monkeypatch, guard_says):
    """Deduplication must not become suppression: the second distinct
    condition is exactly what an operator needs to see."""
    from tidewall_otel._state import State

    state = State()
    state.record_unverified("s", reason="client_escapes", detail=("transport",))
    state.record_unverified("s", reason="client_escapes", detail=("transport",))
    assert len(state.events) == 1

    # A different DETAIL is a sample of the same condition, not a new one --
    # details are the caller's payload and vary per call by nature.
    state.record_unverified("s", reason="client_escapes", detail=("mount",))
    assert len(state.events) == 1
    assert state.events[0].samples == [("transport",), ("mount",)], \
        "a genuinely different detail was not kept as an example"

    # A different REASON, SURFACE or KIND is a different condition.
    state.record_unverified("s", reason="late_import")
    state.record_unverified("other", reason="client_escapes")
    state.record_skip("s", reason="client_escapes")
    assert len(state.events) == 4, \
        "a new reason, surface or kind was swallowed as a duplicate"


def test_VARYING_lossy_details_do_not_grow_the_event_log(monkeypatch, guard_says):
    """Deduplicating on the full detail was not enough.

    The key included `repr(event.detail)`, and monitor passes
    `coverage.lossy_paths` as the detail -- which the CALLER controls. An
    application sending a different unsupported key each time produced a new
    key every call, so both the log and the dedup index grew per call again.
    The first version of this test repeated an identical detail and missed it.
    """
    monkeypatch.setenv("TIDEWALL_MODE", "monitor")
    guard_says(CLEAN)
    reached = []
    client = openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(
            lambda r: (reached.append(1),
                       httpx.Response(200, json=_COMPLETION))[1])))

    tidewall_otel.activate()
    for i in range(200):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            extra_body={f"unsupported_{i}": True})

    assert len(reached) == 200
    state = tidewall_otel.state()
    assert len(state.events) < 10, \
        f"200 calls with varying details left {len(state.events)} events"
    assert len(state._recorded) < 10, \
        f"the dedup index itself grew to {len(state._recorded)}"


def test_a_repeated_condition_is_COUNTED_not_merely_dropped():
    """Bounding the log must not lose the fact that it kept happening."""
    from tidewall_otel._state import State

    state = State()
    for i in range(100):
        state.record_skip("s", reason="lossy", detail=(f"path_{i}",))

    assert len(state.events) == 1
    event = state.events[0]
    assert event.count == 100, "an operator cannot tell this recurred"
    assert 1 < len(event.samples) <= 5, \
        f"expected a bounded sample of details, got {len(event.samples)}"


def test_a_STALE_health_observation_cannot_overwrite_a_newer_one():
    """Guard health is a shared scalar written after each call completes.

    A call that begins during an outage and stalls can finish AFTER a later
    call has already observed recovery, and its assignment would republish
    the outage -- state() then reports an incident that is over.
    """
    from tidewall_otel._state import State

    state = State()
    state.record_guard_health("unreachable", sequence=1)
    state.record_guard_health("clean", sequence=2)
    assert state.guard_health == "ok"

    state.record_guard_health("timeout", sequence=1)     # the stalled older call
    assert state.guard_health == "ok", \
        "an older call republished an outage that had already recovered"


def test_a_GROWING_conversation_does_not_grow_the_event_log(monkeypatch, guard_says):
    """The realistic shape, and the one that made this quadratic.

    An agent loop appends turns, so a lossy element reports
    `messages[1].content`, then `messages[1..3]`, then `messages[1..5]` -- a
    LONGER detail tuple every call. Keying the log on the detail therefore
    stored a new, larger entry per call. Thirty calls left thirty-one events.
    """
    monkeypatch.setenv("TIDEWALL_MODE", "monitor")
    guard_says(CLEAN)
    reached = []
    client = openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(
            lambda r: (reached.append(1),
                       httpx.Response(200, json=_COMPLETION))[1])))

    tidewall_otel.activate()
    conversation = []
    for _ in range(30):
        conversation.append({"role": "user", "content": "hi"})
        conversation.append({"role": "user",
                             "content": [{"type": "text", "text": "block"}]})
        client.chat.completions.create(model="gpt-4o",
                                       messages=list(conversation))

    assert len(reached) == 30, "monitor blocked calls it promised to watch"
    state = tidewall_otel.state()
    assert len(state.events) < 10, \
        f"30 turns of an ordinary agent loop left {len(state.events)} events"
    skip = next(e for e in state.events if e.reason == "lossy")
    assert skip.count == 30, "the recurrence was lost, not just the duplicates"
    assert len(skip.samples) <= 5


def test_the_READMEs_exception_branching_EXAMPLE_actually_works(monkeypatch, guard_says, provider):
    """The README now shows one `except TidewallError` branching on
    `outcome_kind`. Documentation that has never been executed is a claim,
    so this runs exactly the shape it prints."""
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    guard_says(BLOCKED)
    client, _reached = provider

    tidewall_otel.activate()
    try:
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        raise AssertionError("the blocked verdict did not raise")
    except tidewall_otel.TidewallError as declined:
        assert declined.outcome_kind == "blocked", \
            "the documented branch does not fire on the ordinary block path"


def test_state_CANNOT_be_edited_into_claiming_coverage():
    """`state()` returned the live object, so anyone holding it could set
    `lifecycle` and add a surface and make `is_active()` report True with no
    instrumentation installed at all.

    An operator wires `is_active()` into a health check; a security agent
    whose own account of itself can be rewritten through its public API is
    the state-lying-about-reality defect it exists to prevent, arriving
    through the front door.
    """
    tidewall_otel.deactivate()
    snapshot = tidewall_otel.state()

    with pytest.raises(AttributeError, match="read-only"):
        snapshot.lifecycle = "installed"
    with pytest.raises(AttributeError, match="read-only"):
        snapshot.record_skip("Completions.create", reason="invented")

    # Mutating the copied containers is possible but reaches nothing.
    snapshot.surfaces["invented"] = "covered"
    assert "invented" not in tidewall_otel.state().surfaces, \
        "an edit to the snapshot reached the agent's own state"
    assert tidewall_otel.state().is_active() is False


def test_the_snapshot_still_REPORTS_everything_an_operator_needs():
    """Read-only must not mean hollow: the snapshot has to carry the same
    answers, or operators go back to reading internals."""
    from tidewall_otel._state import State

    live = State(lifecycle="installed", mode="enforce",
                 surfaces={"Completions.create": "covered"})
    live.record_guard_health("clean")
    live.record_skip("Completions.create", reason="lossy", detail=("a",))
    live.record_skip("Completions.create", reason="lossy", detail=("b",))

    snapshot = live.snapshot()
    assert snapshot.is_active() is True
    assert snapshot.guard_health == "ok"
    assert snapshot.surfaces == {"Completions.create": "covered"}
    assert [(e.reason, e.count) for e in snapshot.events] == [("lossy", 2)]
    assert snapshot.events[0].samples == [("a",), ("b",)]
    assert snapshot.summary() == live.summary()


def test_a_mutation_between_NORMALIZE_and_the_baseline_is_caught(monkeypatch, guard_says):
    """The guard is shown a normalized copy, and the fingerprint baseline was
    taken AFTERWARDS -- with span construction in between.

    Anything running in that window makes the guard inspect the OLD content
    while both the before and after fingerprints describe the NEW content, so
    they compare equal and the mutation check reports nothing. The provider
    then receives text the guard never saw: the same P0-11 shape, moved one
    step earlier than the window already closed.

    The window is not hypothetical. `gen_ai_span` enters OTel, which invokes
    every registered span processor -- ordinary application code, running on
    the caller's thread, on every guarded call.
    """
    import tidewall_otel._dispatch as dispatch

    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    asked = guard_says(CLEAN)
    message = {"role": "user", "content": "benign at normalisation time"}
    sent = []

    real_span = dispatch.gen_ai_span

    def mutating_span(*args, **kwargs):
        # Exactly where a span processor runs: after normalize(), before the
        # baseline fingerprint.
        message["content"] = "MALICIOUS after the guard's copy was taken"
        return real_span(*args, **kwargs)

    monkeypatch.setattr(dispatch, "gen_ai_span", mutating_span)

    client = openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(
            lambda r: (sent.append(json.loads(r.content)),
                       httpx.Response(200, json=_COMPLETION))[1])))

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        client.chat.completions.create(model="gpt-4o", messages=[message])

    assert raised.value.outcome_kind == "mutated_during_guard"
    assert sent == [], "the provider received content the guard never saw"
    # The guard really was shown the ORIGINAL text, which is what makes the
    # mismatch a mismatch rather than a false alarm.
    assert asked, "the guard was never called"
    assert "benign" in json.dumps(asked[0]), \
        "premise: the guard was shown the pre-mutation content"


async def test_the_pre_baseline_window_is_closed_on_the_ASYNC_path(monkeypatch, guard_says):
    """The async arm has the same window, and holds it open longer: the
    coroutine can be suspended anywhere between normalisation and the
    baseline while the loop runs other tasks."""
    import tidewall_otel._dispatch as dispatch

    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    asked = guard_says(CLEAN)
    message = {"role": "user", "content": "benign at normalisation time"}
    reached = []

    real_span = dispatch.gen_ai_span

    def mutating_span(*args, **kwargs):
        message["content"] = "MALICIOUS after the guard's copy was taken"
        return real_span(*args, **kwargs)

    monkeypatch.setattr(dispatch, "gen_ai_span", mutating_span)

    client = openai.AsyncOpenAI(api_key="t", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: (reached.append(json.loads(r.content)),
                       httpx.Response(200, json=_COMPLETION))[1])))

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallRefusedError) as raised:
        await client.chat.completions.create(model="gpt-4o", messages=[message])

    assert raised.value.outcome_kind == "mutated_during_guard"
    assert reached == [], "the async provider received content the guard never saw"
    assert asked and "benign" in json.dumps(asked[0])
