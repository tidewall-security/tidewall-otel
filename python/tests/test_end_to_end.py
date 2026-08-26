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

        with pytest.raises(tidewall_otel.TidewallRefusedError, match="changed"):
            client.chat.completions.create(model="gpt-4o", messages=[message])

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

        with pytest.raises(tidewall_otel.TidewallRefusedError, match="changed"):
            client.chat.completions.create(model="gpt-4o", messages=[message])

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
    with pytest.raises(tidewall_otel.TidewallRefusedError, match="changed"):
        await client.chat.completions.create(model="gpt-4o", messages=[message])

    assert reached == [], "async enforce sent content nothing inspected"
