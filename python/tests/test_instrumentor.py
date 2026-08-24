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
    and remove wrappers it does not own."""
    tidewall_otel.activate()
    instrumentor = tidewall_otel._instrumentor_instance
    assert instrumentor is not None
    assert hasattr(instrumentor, "_patched")


def test_instrument_is_IDEMPOTENT_under_the_base_class_gate():
    instrumentor = TidewallInstrumentor()
    instrumentor.instrument()
    first = list(instrumentor._patched)
    instrumentor.instrument()
    assert list(instrumentor._patched) == first
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
    instrumentor._patched = []
    instrumentor._uninstrument()

    assert instrumentor.residuals, "a stuck wrapper was not reported"
    assert any("not-ours" in r for r in instrumentor.residuals)


def test_instrumentation_dependencies_names_the_sdks():
    """The OTel loader reads this to decide whether to load the adapter."""
    deps = " ".join(TidewallInstrumentor().instrumentation_dependencies())
    assert "openai" in deps or "anthropic" in deps
