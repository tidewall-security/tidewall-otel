"""Dispatch: pure decisions and the sync adapter. Task 12b."""

import pytest

from tidewall_otel._config import TidewallConfig
from tidewall_otel._dispatch import (
    BoundCall,
    LossyInputError,
    Proceed,
    Refuse,
    SendToGuard,
    TidewallRefusedError,
    Transform,
    decide_outcome,
    dispatch_outcome_for,
    dispatch_sync,
)
from tidewall_otel._exceptions import TidewallBlockedError, TidewallError
from tidewall_otel._execution import BoundedExecutor, ExecutorSaturated
from tidewall_otel._http import GuardSchemaInvalid, GuardTimeout, GuardUnreachable
from tidewall_otel._manifest import ANTHROPIC_MESSAGES_SYNC, OPENAI_CHAT_SYNC
from tidewall_otel._response import Outcome
import tidewall_otel._dispatch as _dispatch_module
from tidewall_otel._dispatch import _EXCEPTION_OUTCOMES, _FAILURES
from tidewall_otel._state import State

#: DERIVED from production, not retyped. The hand-written copy had ALREADY
#: drifted -- it omitted "incomplete", so removing that kind from `_FAILURES`
#: left both dispatch suites green. A second copy of a set is a copy that can
#: disagree, and this one did.
FAILURE_KINDS = sorted(_FAILURES)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    return TidewallConfig()


@pytest.fixture
def executor():
    ex = BoundedExecutor(max_workers=2, queue_size=4)
    yield ex
    ex.shutdown()


class FakeInstance:
    """Stands in for wrapt's `instance` -- the SDK RESOURCE, whose `_client`
    dispatch reads. Returning the client itself would make every escape test
    pass for the wrong reason."""

    def __init__(self, client=None):
        self._client = client


def minimal(surface):
    if surface.provider == "openai":
        return {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    return {"model": "c", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]}


class RecordingProvider:
    def __init__(self, result="provider-result"):
        self.calls = []
        self._result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._result


class GuardReturning:
    def __init__(self, body):
        self.calls = []
        self._body = body

    def check_raw(self, *, guard_input, **kwargs):
        self.calls.append(guard_input)
        return self._body


class GuardRaising:
    _EXC = {
        "unreachable": GuardUnreachable("refused"),
        "timeout": GuardTimeout("deadline"),
        "saturated": ExecutorSaturated("full"),
        "schema_invalid": GuardSchemaInvalid("not json"),
        "invariant_violated": ZeroDivisionError("boom"),
    }

    def __init__(self, kind):
        self.calls = []
        self._exc = self._EXC[kind]

    def check_raw(self, *, guard_input, **kwargs):
        self.calls.append(guard_input)
        raise self._exc


def clean_body():
    return {"request_id": "r", "request_time": "t", "summary": "",
            "result": {"blocked": False, "transformed": False, "policy": "d"}}


# -- the exception table ---------------------------------------------------

@pytest.mark.parametrize("kind", FAILURE_KINDS[:-1])
def test_each_guard_exception_maps_to_its_outcome_kind(kind):
    assert dispatch_outcome_for(GuardRaising._EXC[kind]).kind == kind


def test_an_unexpected_exception_is_invariant_violated_NOT_swallowed():
    assert dispatch_outcome_for(ZeroDivisionError()).kind == "invariant_violated"


# -- the pure decisions ----------------------------------------------------

def test_a_transform_preserves_every_untouched_provider_kwarg(config):
    """A classified response carries only what the guard rewrote, so without
    the original bound call a transform would reconstruct arguments it never
    saw."""
    call = BoundCall(None, None, (), {"messages": [], "model": "gpt-4o",
                                      "temperature": 0.2, "seed": 7}, {})
    decision = decide_outcome(
        OPENAI_CHAT_SYNC, call, SendToGuard({}),
        Outcome("transformed", guard_output={"messages": [
            {"role": "user", "content": "clean"}]}), config)

    assert isinstance(decision, Transform)

    # DERIVED, not named. Listing `temperature` and `seed` checks the two
    # kwargs the author happened to think of; a kwarg added to the fixture
    # later, or dropped by the transform, would go unnoticed under a name
    # promising EVERY untouched kwarg.
    untouched = {key: value for key, value in call.kwargs.items()
                 if key != "messages"}
    assert {key: decision.kwargs.get(key) for key in untouched} == untouched, (
        f"a kwarg the guard never saw was altered: {decision.kwargs}")
    assert set(decision.kwargs) == set(call.kwargs), (
        "the transform added or dropped a kwarg")
    assert decision.kwargs["messages"][0]["content"] == "clean"


def test_an_anthropic_transform_writes_the_system_prompt_BACK_TO_system(config):
    call = BoundCall(None, None, (), {"messages": [{"role": "user", "content": "hi"}],
                                      "system": "be careful", "model": "c"}, {})
    decision = decide_outcome(
        ANTHROPIC_MESSAGES_SYNC, call, SendToGuard({}),
        Outcome("transformed", guard_output={"messages": [
            {"role": "system", "content": "cleaned"},
            {"role": "user", "content": "hi"}]}), config)

    assert decision.kwargs["system"] == "cleaned"
    assert all(m["role"] != "system" for m in decision.kwargs["messages"])


@pytest.mark.parametrize("kind", FAILURE_KINDS)
def test_enforce_REFUSES_every_failure_kind(config, kind):
    call = BoundCall(None, None, (), {}, {})
    decision = decide_outcome(OPENAI_CHAT_SYNC, call, SendToGuard({}),
                              Outcome(kind), config)
    assert isinstance(decision, Refuse)


@pytest.mark.parametrize("mode", ["monitor", "dry-run"])
@pytest.mark.parametrize("kind", FAILURE_KINDS)
def test_monitor_and_dry_run_PROCEED_on_every_failure_kind(monkeypatch, mode, kind):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", mode)
    call = BoundCall(None, None, (), {"a": 1}, {})
    decision = decide_outcome(OPENAI_CHAT_SYNC, call, SendToGuard({}),
                              Outcome(kind), TidewallConfig())
    assert isinstance(decision, Proceed)
    assert decision.kwargs == {"a": 1}


# -- the sync adapter ------------------------------------------------------

@pytest.mark.parametrize("kind", FAILURE_KINDS)
def test_enforce_NEVER_INVOKES_the_provider_on_any_failure(config, executor, kind):
    """Asserted on INVOCATION. The agent proceeds to the provider whenever
    guard checking returns no decision, so a test reading the return value
    cannot see the bug."""
    provider = RecordingProvider()
    guard = GuardRaising(kind)
    with pytest.raises(TidewallError) as raised:
        dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (),
                      minimal(OPENAI_CHAT_SYNC), config, guard, executor)
    assert provider.calls == [], f"contacted the provider on {kind}"
    # `TidewallError` alone is a FALSE GREEN: `LossyInputError` is one, so a
    # regression classifying this fixture as lossy before the guard is ever
    # called would satisfy both assertions while none of the failure mapping
    # under test runs. Name the reason, and prove the guard was reached.
    assert guard.calls, f"the guard was never called on {kind}"
    assert raised.value.outcome_kind == kind, \
        f"refused as {raised.value.outcome_kind}, not {kind}"


@pytest.mark.parametrize("mode", ["monitor", "dry-run"])
@pytest.mark.parametrize("kind", FAILURE_KINDS)
def test_monitor_and_dry_run_ALWAYS_invoke_and_record(monkeypatch, executor, mode, kind):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", mode)
    provider = RecordingProvider()
    result = dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (),
                           minimal(OPENAI_CHAT_SYNC), TidewallConfig(),
                           GuardRaising(kind), executor)
    assert provider.calls, f"did not reach the provider in {mode}"
    assert result == "provider-result"


def test_a_clean_call_reaches_the_provider_and_the_guard(config, executor):
    provider, guard = RecordingProvider(), GuardReturning(clean_body())
    result = dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (),
                           minimal(OPENAI_CHAT_SYNC), config, guard, executor)
    assert result == "provider-result"
    assert guard.calls and sorted(guard.calls[0]) == ["messages", "tools"]


def test_a_typed_content_call_in_enforce_reaches_NEITHER_guard_NOR_provider(
        config, executor):
    """The dispatch-level evidence for the lossy decision."""
    provider, guard = RecordingProvider(), GuardReturning(clean_body())
    typed = {"model": "gpt-4o", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}

    with pytest.raises(LossyInputError):
        dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (), typed,
                      config, guard, executor)

    assert guard.calls == [], "sent lossy input to the guard"
    assert provider.calls == [], "called the provider with uninspected content"


def test_a_nonempty_extra_body_in_enforce_is_REFUSED(config, executor):
    """O-11 at dispatch: the bypass is refused before either side is reached."""
    provider, guard = RecordingProvider(), GuardReturning(clean_body())
    call = {**minimal(OPENAI_CHAT_SYNC),
            "extra_body": {"messages": [{"role": "user", "content": "EVIL"}]}}

    with pytest.raises(LossyInputError) as caught:
        dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (), call,
                      config, guard, executor)

    assert "extra_body" in str(caught.value)
    assert guard.calls == [] and provider.calls == []


def test_a_lossy_call_in_MONITOR_is_recorded_AND_reaches_the_provider(
        monkeypatch, executor):
    """Monitor exists to produce evidence, so the one call worth recording
    must not be the one that records nothing."""
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "monitor")
    provider, guard = RecordingProvider(), GuardReturning(clean_body())
    state = State(lifecycle="installed", mode="monitor")
    typed = {"model": "gpt-4o", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}

    dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (), typed,
                  TidewallConfig(), guard, executor, state=state)

    assert provider.calls, "monitor must not block the call"
    assert guard.calls == [], "lossy input must not be sent even in monitor"
    assert any("lossy" in str(event) for event in state.events), state.events


def test_dispatch_INVOKES_check_raw_not_the_legacy_check(config, executor):
    """The legacy check returns a GuardResult and swallows transport failures
    into None -- the fail-open this replaces."""
    calls = []

    class Recording:
        def check_raw(self, *, guard_input, **kwargs):
            calls.append("check_raw")
            return clean_body()

        def check(self, *, guard_input, **kwargs):
            raise AssertionError("dispatch called the legacy fail-open check")

    dispatch_sync(OPENAI_CHAT_SYNC, RecordingProvider(), FakeInstance(), (),
                  minimal(OPENAI_CHAT_SYNC), config, Recording(), executor)
    assert calls == ["check_raw"]


def test_a_client_escape_records_UNVERIFIED(config, executor):
    """Out of the threat model, but never unnoticed."""
    import httpx
    import openai

    escaping = openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    state = State(lifecycle="installed", mode="enforce",
                  surfaces={OPENAI_CHAT_SYNC.attribute: "covered"})

    dispatch_sync(OPENAI_CHAT_SYNC, RecordingProvider(), FakeInstance(escaping),
                  (), minimal(OPENAI_CHAT_SYNC), config,
                  GuardReturning(clean_body()), executor, state=state)

    assert state.surfaces[OPENAI_CHAT_SYNC.attribute] == "unverified"
    assert state.is_active() is False


def blocked_body():
    return {"request_id": "r", "request_time": "t", "summary": "policy hit",
            "result": {"blocked": True, "transformed": False, "policy": "d"}}


def test_a_BLOCKED_verdict_in_enforce_never_reaches_the_provider(config, executor):
    """The verdict the product exists to deliver, asserted on INVOCATION.

    Mutation-testing found this untested: every failure KIND was covered, and
    the ordinary blocked path -- the guard working correctly and saying no --
    was not.
    """
    provider = RecordingProvider()
    guard = GuardReturning(blocked_body())
    with pytest.raises(TidewallBlockedError):
        dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (),
                      minimal(OPENAI_CHAT_SYNC), config, guard, executor)
    assert provider.calls == [], "a blocked call reached the provider"
    assert guard.calls, "nothing was blocked -- the guard was never called"


@pytest.mark.parametrize("mode", ["monitor", "dry-run"])
def test_a_BLOCKED_verdict_outside_enforce_proceeds(monkeypatch, executor, mode):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", mode)
    provider = RecordingProvider()
    result = dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (),
                           minimal(OPENAI_CHAT_SYNC), TidewallConfig(),
                           GuardReturning(blocked_body()), executor)
    assert provider.calls and result == "provider-result"


def test_a_TRANSFORMED_verdict_rewrites_what_reaches_the_provider(config, executor):
    """The transform must actually change the outgoing call, not merely be
    classified."""
    provider = RecordingProvider()
    transformed = {"request_id": "r", "request_time": "t", "summary": "",
                   "result": {"blocked": False, "transformed": True, "policy": "d",
                              "guard_output": {"messages": [
                                  {"role": "user", "content": "REDACTED"}]}}}

    dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (),
                  minimal(OPENAI_CHAT_SYNC), config,
                  GuardReturning(transformed), executor)

    _args, kwargs = provider.calls[0]
    assert kwargs["messages"][0]["content"] == "REDACTED"
    assert kwargs["model"] == "gpt-4o", "an untouched kwarg was lost"


def test_ONE_except_clause_catches_every_refusal():
    """A blocked verdict and a refused call are the same event to the
    application. Two unrelated exception hierarchies would let one escape a
    handler written for the other -- which is how this was found."""
    from tidewall_otel._dispatch import LossyInputError, TidewallRefusedError
    from tidewall_otel._exceptions import TidewallBlockedError, TidewallError

    for cls in (TidewallBlockedError, TidewallRefusedError, LossyInputError):
        assert issubclass(cls, TidewallError), cls


# -- mode branches: correct, but previously unasserted ---------------------
# A mechanical mutation sweep -- one that generates mutations from the source
# rather than from the author's awareness -- found these branches unconstrained.
# The behaviour was right; nothing held it there.

def test_dry_run_makes_NO_guard_call_at_all(monkeypatch, executor):
    """dry-run's contract is that no request leaves the process. Asserting
    only that the provider was reached cannot distinguish it from monitor,
    because both proceed."""
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "dry-run")

    provider, guard = RecordingProvider(), GuardReturning(clean_body())
    result = dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (),
                           minimal(OPENAI_CHAT_SYNC), TidewallConfig(),
                           guard, executor)

    assert guard.calls == [], "dry-run contacted the guard"
    assert provider.calls and result == "provider-result"


def transformed_body():
    return {"request_id": "r", "request_time": "t", "summary": "",
            "result": {"blocked": False, "transformed": True, "policy": "d",
                       "guard_output": {"messages": [
                           {"role": "user", "content": "REDACTED"}]}}}


@pytest.mark.parametrize("mode", ["monitor", "dry-run"])
def test_a_transform_is_NOT_APPLIED_outside_enforce(monkeypatch, executor, mode):
    """monitor observes; it does not rewrite. Applying a transform there would
    silently change what the application sends while the operator believes the
    mode is read-only -- and the failure-kind tests cannot see it, because
    those never produce a transformed verdict."""
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", mode)

    provider = RecordingProvider()
    dispatch_sync(OPENAI_CHAT_SYNC, provider, FakeInstance(), (),
                  {"model": "gpt-4o", "messages": [{"role": "user", "content": "ORIGINAL"}]},
                  TidewallConfig(), GuardReturning(transformed_body()), executor)

    _args, kwargs = provider.calls[0]
    assert kwargs["messages"][0]["content"] == "ORIGINAL", (
        f"{mode} rewrote the request it was only meant to observe"
    )


def test_every_FAILURE_KIND_can_actually_be_produced():
    """A failure kind nothing emits is a protection that is not operating.

    `_FAILURES` carried "incomplete", which no code path constructs. The
    dispatch tests retyped the set by hand and omitted it, so the dead entry
    was invisible from both sides -- the test list looked like it had drifted
    from production when production had drifted from reality.

    Derived by scanning the source for constructed kinds, so a kind added to
    `_FAILURES` without a producer fails here rather than sitting inert.
    """
    import pathlib
    import re

    src = pathlib.Path(_dispatch_module.__file__).parent
    produced = set()
    for path in src.glob("*.py"):
        produced |= set(re.findall(r'kind=["\']([a-z_]+)["\']', path.read_text()))
    produced |= set(_EXCEPTION_OUTCOMES.values())

    unproducible = _FAILURES - produced
    assert not unproducible, (
        f"failure kinds nothing can emit: {sorted(unproducible)}")
