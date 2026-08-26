"""Activation semantics. Task 11 of the P0 remediation plan.

Scope is strictly ACTIVATION: what happens at activate() time, and which
startup configurations are rejected. Nothing here invokes a wrapped provider,
so nothing here asserts whether one ran -- the runtime failure matrix belongs
to dispatch, where the wrappers consume coverage and mode decisions.
"""

import pytest

import tidewall_otel
from tidewall_otel._config import TidewallConfig
from tidewall_otel._exceptions import TidewallConfigError
from tidewall_otel._state import State


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


@pytest.mark.parametrize("mode", ["monitor", "dry-run"])
def test_block_is_REJECTED_AT_STARTUP_in_monitor_and_dry_run(monkeypatch, mode):
    """These pairs cannot be exercised as runtime cases at all: the run raises
    before any call is dispatched."""
    monkeypatch.setenv("TIDEWALL_MODE", mode)
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "block")
    with pytest.raises(ValueError, match="block"):
        TidewallConfig()


@pytest.mark.parametrize("mode,on_failure", [
    ("enforce", "exit"), ("enforce", "block"), ("enforce", "disable"),
    ("monitor", "exit"), ("monitor", "disable"),
    ("dry-run", "exit"), ("dry-run", "disable"),
])
def test_each_VALID_pair_activates(monkeypatch, mode, on_failure):
    """The seven valid pairs; the two invalid ones are startup rejections
    above, not runtime cases."""
    monkeypatch.setenv("TIDEWALL_MODE", mode)
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", on_failure)

    tidewall_otel.activate()
    state = tidewall_otel.state()
    assert state.mode == mode
    assert state.lifecycle == "installed"


def test_activation_is_IDEMPOTENT(monkeypatch):
    tidewall_otel.activate()
    first = tidewall_otel.state()
    tidewall_otel.activate()
    # Equality, not identity: `state()` hands out a fresh read-only snapshot
    # each call. The claim is that activating twice CHANGES nothing, which
    # identity only ever tested by proxy.
    assert tidewall_otel.state() == first


def test_deactivation_returns_the_lifecycle_to_removed(monkeypatch):
    tidewall_otel.activate()
    assert tidewall_otel.state().lifecycle == "installed"
    tidewall_otel.deactivate()
    assert tidewall_otel.state().lifecycle == "removed"
    assert tidewall_otel.state().is_active() is False


def test_state_is_QUERYABLE_before_activation():
    """An operator asking 'is it on?' before activation must get an answer,
    not an exception or None."""
    state = tidewall_otel.state()
    assert isinstance(state, State)
    assert state.lifecycle in ("uninstalled", "removed")
    assert state.is_active() is False


def test_a_config_error_does_NOT_silently_fail_open(monkeypatch):
    """Logging an error and continuing unguarded, while the application
    believes it is protected, is the fail-open this programme removes.

    In enforce the activation failure policy decides; the default `exit` means
    the process does not continue believing it is guarded.
    """
    monkeypatch.delenv("TIDEWALL_BASE_URL", raising=False)
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "exit")

    with pytest.raises(TidewallConfigError, match="TIDEWALL_BASE_URL"):
        tidewall_otel.activate()


def test_disable_leaves_the_application_running_UNGUARDED_and_says_so(monkeypatch):
    """`disable` is the honest fail-open: the caller chose it, and the state
    must say the agent is not active rather than pretend otherwise."""
    monkeypatch.delenv("TIDEWALL_BASE_URL", raising=False)
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "disable")

    tidewall_otel.activate()                    # must NOT raise
    state = tidewall_otel.state()
    assert state.is_active() is False
    assert state.lifecycle == "uninstalled"


def test_block_installs_REFUSERS_when_coverage_cannot_be_verified(monkeypatch):
    """`block` is only honoured when refusal coverage is verified at every
    boundary; the surfaces become `refusing` rather than silently covered."""
    monkeypatch.delenv("TIDEWALL_BASE_URL", raising=False)
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "block")

    tidewall_otel.activate()                    # must NOT raise
    state = tidewall_otel.state()
    assert state.is_active() is False
    # NON-EMPTY first: `all()` over an empty dict is vacuously true, so
    # without this the assertion passes when the block branch never runs at
    # all -- which is exactly what mutation-testing found.
    assert state.surfaces, "no surfaces were marked refusing"
    assert all(d == "refusing" for d in state.surfaces.values()), state.surfaces


def test_is_active_DELEGATES_to_the_state(monkeypatch):
    """A boolean maintained beside the dimensions can disagree with them, and
    did. Nothing called the public helper, so a hardcoded True survived."""
    assert tidewall_otel.is_active() is False
    tidewall_otel.activate()
    assert tidewall_otel.is_active() is tidewall_otel.state().is_active()
    tidewall_otel.deactivate()
    assert tidewall_otel.is_active() is False


def test_the_summary_explains_WHY_it_is_inactive(monkeypatch):
    """`active: false` without a reason is not actionable."""
    monkeypatch.delenv("TIDEWALL_TOKEN", raising=False)
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "disable")
    tidewall_otel.activate()

    summary = tidewall_otel.state().summary()
    assert "active=False" in summary
    assert "lifecycle=" in summary and "mode=" in summary


def test_deactivation_restores_the_EXACT_original_signature():
    """Unwrapping on `__wrapped__` alone strips a layer the SDK itself put
    there, leaving a callable with a different signature that still looks
    plausible -- it surfaced as unrelated manifest tests failing several
    files later, not as an activation failure."""
    import inspect

    from openai.resources.chat.completions.completions import Completions

    before = inspect.getattr_static(Completions, "create")
    before_params = set(inspect.signature(before).parameters)

    tidewall_otel.activate()
    tidewall_otel.deactivate()

    after = inspect.getattr_static(Completions, "create")
    assert after is before, "deactivation did not restore the original object"
    assert set(inspect.signature(after).parameters) == before_params


def test_refusers_are_REMOVED_by_deactivation(monkeypatch):
    """Installing refusers outside instrument() leaves uninstrument() gated
    off, so deactivate() silently does nothing and they stay on the SDK for
    the rest of the process -- which showed up as unrelated tests in another
    file failing, never as an activation failure.

    monkeypatch, not os.environ: mutating the environment directly defeats
    this module's autouse cleanup and pollutes every later test.
    """
    import inspect

    from openai.resources.chat.completions.completions import Completions

    before = inspect.getattr_static(Completions, "create")

    monkeypatch.delenv("TIDEWALL_BASE_URL", raising=False)
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "block")
    tidewall_otel.activate()
    assert inspect.getattr_static(Completions, "create") is not before

    tidewall_otel.deactivate()
    assert inspect.getattr_static(Completions, "create") is before
