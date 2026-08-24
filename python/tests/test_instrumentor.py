"""The OTel adapter. Task 14 of the P0 remediation plan."""

import inspect

import pytest

import tidewall_otel
from tidewall_otel._instrumentor import TidewallInstrumentor


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


def test_the_adapter_does_NOT_call_activate():
    """Mutual recursion: activate() constructs the instrumentor and calls
    instrument(); if _instrument called activate() back, `opentelemetry-instrument`
    would recurse until the stack died -- and only under the OTel entry point,
    never in the direct-call path most tests exercise."""
    source = inspect.getsource(TidewallInstrumentor._instrument)
    assert "activate(" not in source, source


def test_activate_does_NOT_call_the_adapter_hook():
    """The other direction of the same loop."""
    source = inspect.getsource(tidewall_otel.activate)
    assert "_instrument(" not in source, source


def test_the_adapter_is_a_thin_shim_over_the_manager():
    """Patch bookkeeping belongs to the PatchManager, which compares before
    writing. An adapter keeping its own parallel record would disagree with it
    and remove wrappers it does not own.

    The earlier version of this test asserted `hasattr(instrumentor,
    "_patched")` -- it REQUIRED the exact parallel bookkeeping its own
    docstring forbids, and asserted nothing about the manager. It therefore
    passed while activation used neither the manager nor the executor.
    """
    tidewall_otel.activate()
    instrumentor = tidewall_otel._instrumentor_instance
    assert instrumentor is not None

    assert not hasattr(instrumentor, "_patched"), (
        "the adapter is keeping a parallel patch record again"
    )
    assert instrumentor._manager is not None, "activation bypassed the PatchManager"
    assert instrumentor._executor is not None, "activation created no bounded executor"

    # ONE state object, not two. The wrappers were handed `instrumentor.state`;
    # if `state()` returns a different instance, every runtime downgrade they
    # record is invisible to the caller.
    assert tidewall_otel.state() is instrumentor.state

    # And the manager really holds the boundaries, so removal can compare.
    installed = {e.attribute for e in instrumentor._manager.journal if e.kind == "patch"}
    assert installed, "the manager installed nothing"


def test_instrument_is_IDEMPOTENT_under_the_base_class_gate():
    instrumentor = TidewallInstrumentor()
    instrumentor.instrument()
    first = [(e.module, e.attribute) for e in instrumentor._manager.journal]
    instrumentor.instrument()
    assert [(e.module, e.attribute) for e in instrumentor._manager.journal] == first
    instrumentor.uninstrument()


def test_uninstrument_reports_STUCK_rather_than_claiming_removal(monkeypatch):
    """A wrapper that could not be removed is not the same as one that was.

    Reporting it as uninstrumented leaves another agent's SDK carrying our
    wrapper while the state says we are gone -- the removal-side twin of
    reporting enforcement while unguarded.
    """
    from tidewall_otel._manager import PatchManager, RemovalOutcome

    manager = PatchManager()
    monkeypatch.setattr(PatchManager, "remove",
                        lambda self: {("m", "a"): RemovalOutcome.NOT_OURS})

    instrumentor = TidewallInstrumentor()
    instrumentor._manager = manager
    instrumentor._uninstrument()

    assert instrumentor.residuals, "a stuck wrapper was not reported"
    assert any("not-ours" in r for r in instrumentor.residuals)


def test_instrumentation_dependencies_names_the_sdks():
    """The OTel loader reads this to decide whether to load the adapter."""
    deps = " ".join(TidewallInstrumentor().instrumentation_dependencies())
    assert "openai" in deps or "anthropic" in deps


def test_deactivation_PRESERVES_a_wrapper_installed_after_us(monkeypatch):
    """Finding 3. Ownership is an identity comparison, not a type test.

    The removed implementation unwrapped anything that was a
    `wrapt.FunctionWrapper` carrying `__wrapped__`. A wrapper another agent
    installs AFTER us satisfies that exactly as well as ours does, so
    deactivation deleted THEIR wrapper and restored the Tidewall layer
    underneath -- the inverse of the intent, beside a comment claiming "ONLY
    unwrap OUR wrapper".

    Here the foreign wrapper is installed after activation, so the current
    attribute is no longer the object the manager wrote.
    """
    import sys
    import types

    import wrapt

    from tidewall_otel._manager import PatchManager, RemovalOutcome

    module = types.ModuleType("foreign_sdk")

    def original():
        return "original"

    module.Target = type("Target", (), {"create": staticmethod(original)})
    monkeypatch.setitem(sys.modules, "foreign_sdk", module)

    manager = PatchManager()
    manager.install("foreign_sdk", "Target.create",
                    lambda w, i, a, k: w(*a, **k))
    ours = inspect.getattr_static(module.Target, "create")

    # SOMEONE ELSE wraps the method afterwards -- the last writer wins.
    def foreign(wrapped, instance, args, kwargs):
        return "foreign"

    wrapt.wrap_function_wrapper(module, "Target.create", foreign)
    theirs = inspect.getattr_static(module.Target, "create")
    assert theirs is not ours

    outcomes = manager.remove()

    assert outcomes[("foreign_sdk", "Target.create")] is RemovalOutcome.NOT_OURS
    assert inspect.getattr_static(module.Target, "create") is theirs, (
        "deleted a wrapper installed after ours"
    )


def test_deactivation_reports_a_preserved_foreign_wrapper_as_a_RESIDUAL(monkeypatch):
    """Preserving it is right; going quiet about it is not.

    Tidewall's wrapper is still on that SDK, underneath theirs. State that
    says `removed` with no residual would be the removal-side twin of
    reporting enforcement while unguarded.
    """
    import sys
    import types

    import wrapt

    from tidewall_otel._manager import PatchManager

    module = types.ModuleType("foreign_sdk2")
    module.Target = type("Target", (), {"create": staticmethod(lambda: "original")})
    monkeypatch.setitem(sys.modules, "foreign_sdk2", module)

    manager = PatchManager()
    manager.install("foreign_sdk2", "Target.create", lambda w, i, a, k: w(*a, **k))
    wrapt.wrap_function_wrapper(module, "Target.create",
                                lambda w, i, a, k: "foreign")

    instrumentor = TidewallInstrumentor()
    instrumentor._manager = manager
    instrumentor._uninstrument()

    assert instrumentor.residuals, "a preserved foreign wrapper was not reported"
    assert any("not-ours" in r for r in instrumentor.residuals), instrumentor.residuals
