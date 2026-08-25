"""The OTel adapter. Task 14 of the P0 remediation plan."""

import dataclasses
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


def _calls_named(func, name):
    """Every call to `name` in `func`, by AST rather than by substring.

    The substring version failed the moment a COMMENT in `_instrument`
    mentioned `activate()` while calling no such thing -- it could not tell a
    call from prose, so documenting the invariant broke the test guarding it.
    It would equally have missed `getattr(tidewall_otel, "acti" + "vate")()`,
    but that is not the failure mode; an honest mention is.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id == name:
                yield node.lineno
            elif isinstance(callee, ast.Attribute) and callee.attr == name:
                yield node.lineno


def test_the_adapter_does_NOT_call_activate():
    """Mutual recursion: activate() constructs the instrumentor and calls
    instrument(); if _instrument called activate() back, `opentelemetry-instrument`
    would recurse until the stack died -- and only under the OTel entry point,
    never in the direct-call path most tests exercise."""
    found = list(_calls_named(TidewallInstrumentor._instrument, "activate"))
    assert not found, f"_instrument calls activate() at line(s) {found}"


def test_activate_does_NOT_call_the_adapter_hook():
    """The other direction of the same loop."""
    found = list(_calls_named(tidewall_otel.activate, "_instrument"))
    assert not found, f"activate calls _instrument() at line(s) {found}"


def test_the_recursion_detector_CATCHES_a_real_call():
    """Known-positive. The detector must still see a call it should reject --
    otherwise switching from substring to AST could have made it vacuous."""
    def activate():
        return None

    def calls_it():
        activate()                      # a REAL call to a real name

    def calls_it_via_attribute():
        tidewall_otel.activate()

    def only_mentions_it():
        """Prose about activate() that calls nothing."""
        return "activate("

    assert list(_calls_named(calls_it, "activate"))
    assert list(_calls_named(calls_it_via_attribute, "activate"))
    assert not list(_calls_named(only_mentions_it, "activate"))


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


def test_a_not_yet_imported_SDK_is_patched_when_it_ARRIVES(tmp_path, monkeypatch):
    """Round 6's P1. The design requires a journal-owned meta-path finder so a
    module imported AFTER activation is still patched.

    `PatchManager` implemented it and its unit tests passed, but `_instrument`
    said `continue` and never called `register_surface` or `install_finder` --
    the requirement was built, tested in isolation, and left unreachable.

    The plan records that v1 dropped this same requirement and then reported
    §5 covered. Leaving the manager's implementation unwired dropped it again
    one layer along, under a green suite. That is rule 18 with the roles
    reversed: not a caller wired wrongly, but a callee never called.
    """
    import sys

    import tidewall_otel._instrumentor as instrumentor_module
    import tidewall_otel._manifest as manifest_module
    from tidewall_otel._config import TidewallConfig

    (tmp_path / "late_sdk.py").write_text(
        "class Target:\n"
        "    def create(self, **kwargs):\n"
        "        return 'original'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "late_sdk", raising=False)

    surface = dataclasses.replace(manifest_module.OPENAI_CHAT_SYNC,
                                  module="late_sdk", attribute="Target.create")
    # `_instrument` imports SURFACES from the manifest INSIDE the method, so
    # the name to patch is the manifest's, not the instrumentor's.
    monkeypatch.setattr(manifest_module, "SURFACES", (surface,))
    monkeypatch.setattr(instrumentor_module, "_sdk_available",
                        lambda module: module != "late_sdk")

    instrumentor = instrumentor_module.TidewallInstrumentor()
    instrumentor._instrument(config=TidewallConfig(
        base_url="https://guard.example", token="t"))
    try:
        module = __import__("late_sdk")
        assert getattr(inspect.getattr_static(module.Target, "create"),
                       "__tidewall_wrapper__", False), (
            "a module imported after activation was left unguarded")
    finally:
        instrumentor._uninstrument()


def test_an_ABSENT_sdk_does_not_make_a_guarded_app_report_inactive(monkeypatch):
    """The other direction, and the one that condemns everyone if wrong.

    `is_active()` is a universal claim over the boundaries PRESENT. A provider
    that is not installed presents none. Recording its surfaces as `uncovered`
    or `deferred` would make `is_active()` False for an application that
    installed one provider and is fully guarded on it -- the same
    over-correction the escape detector had to avoid.
    """
    import tidewall_otel._instrumentor as instrumentor_module

    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    # anthropic is not installed in this hypothetical deployment
    monkeypatch.setattr(instrumentor_module, "_sdk_available",
                        lambda module: "anthropic" not in module)

    tidewall_otel.activate()
    state = tidewall_otel.state()

    assert state.surfaces, "no boundary was recorded at all"
    assert not any("Messages" in name for name in state.surfaces), (
        f"an absent provider's surfaces were recorded: {state.surfaces}")
    assert state.is_active() is True, (
        f"a fully guarded app reported inactive: {state.surfaces}")


def test_a_FAILED_finder_install_is_reported_as_uncovered(monkeypatch):
    """Round 7's other P1, and a correction to my own reasoning.

    I argued that surfaces in a not-yet-imported module should stay out of
    `state.surfaces` because `is_active()` is a claim over boundaries PRESENT.
    That holds only while the finder installs. When it FAILS, the agent knows
    a boundary that may arrive can never be patched -- and it was recording
    exactly that in `manager.dispositions` while `is_active()` returned True.

    The design is explicit: if the finder cannot be installed, in-scope
    surfaces from not-yet-imported modules are `uncovered`.
    """
    import tidewall_otel._instrumentor as instrumentor_module
    from tidewall_otel._manager import PatchManager

    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setattr(instrumentor_module, "_sdk_available",
                        lambda module: "anthropic" not in module)

    def cannot_install(self, modules):
        for name in frozenset(modules):
            self.dispositions[name] = "uncovered"

    monkeypatch.setattr(PatchManager, "install_finder", cannot_install)

    tidewall_otel.activate()
    state = tidewall_otel.state()

    # DERIVE the expectation, do not name one surface. Anthropic contributes
    # two boundaries here; asserting only `Messages.create` let a mutation
    # narrowing the production loop to `attributes[:1]` pass, so the test
    # named as the finder-failure guarantee did not hold the universal
    # requirement in its own docstring. Eighth quantifier defect of the
    # session, and the second in a test I wrote to close one.
    from tidewall_otel._manifest import SURFACES

    deferred = {surface.attribute for surface in SURFACES
                if "anthropic" in surface.module}
    assert len(deferred) > 1, (
        "this test needs a provider with MORE than one surface to be able to "
        "distinguish 'all' from 'the first'")

    uncovered = {name for name, disposition in state.surfaces.items()
                 if disposition == "uncovered"}
    assert deferred <= uncovered, (
        f"deferred surfaces not marked uncovered: {sorted(deferred - uncovered)}")

    reported = {event.surface for event in state.events
                if event.reason == "finder_not_installed"}
    assert deferred <= reported, (
        f"deferred surfaces with no event: {sorted(deferred - reported)}")

    assert state.is_active() is False, (
        "reported active while knowing a boundary can never be patched")


def test_a_STUCK_removal_keeps_the_manager_so_deactivate_can_RETRY(monkeypatch):
    """Adversarial review finding 3, and it is not merely untidy.

    `remove()` deliberately retains `not_ours` and `errored` entries so
    removal can be retried once the conflicting wrapper goes. `_uninstrument`
    then discarded the manager unconditionally -- the only object holding
    their `pre_install_identity` -- which made that retry unreachable from
    the public API.

    The consequence: when the foreign layer is later removed, OUR wrapper
    becomes the live attribute again, with nothing left able to take it off.
    Permanent stale instrumentation on somebody else's SDK, while the
    residual report says only that we noticed.
    """
    import sys
    import types

    import wrapt

    from tidewall_otel._manager import PatchManager

    module = types.ModuleType("stuck_sdk")
    module.Target = type("Target", (), {"create": staticmethod(lambda: "original")})
    monkeypatch.setitem(sys.modules, "stuck_sdk", module)

    manager = PatchManager()
    manager.install("stuck_sdk", "Target.create", lambda w, i, a, k: w(*a, **k))

    # Another agent wraps on top, so removal must decline.
    wrapt.wrap_function_wrapper(module, "Target.create",
                                lambda w, i, a, k: "foreign")
    theirs = inspect.getattr_static(module.Target, "create")

    instrumentor = TidewallInstrumentor()
    instrumentor._manager = manager
    instrumentor._uninstrument()

    assert instrumentor.residuals, "a stuck entry was not reported"
    assert instrumentor._manager is manager, (
        "the manager was discarded, so the retained entry can never be removed")
    assert inspect.getattr_static(module.Target, "create") is theirs

    # The foreign layer goes. Restore EXACTLY the object the manager
    # installed -- re-wrapping it in a fresh `staticmethod` would change its
    # identity, and removal compares identity, so the retry would decline
    # again for a different reason and the test would prove nothing.
    pristine = manager.journal[0].pre_install_identity
    setattr(module.Target, "create", manager.journal[0].installed)

    instrumentor._uninstrument()

    # Assert IDENTITY, not the call result. Our wrapper is a pass-through, so
    # `create()` returns "original" whether or not it was ever removed -- the
    # first version of this assertion passed in both worlds.
    assert inspect.getattr_static(module.Target, "create") is pristine, (
        "the retry did not restore the original attribute")
    assert manager.journal == [], "the journal was not discharged"
    assert instrumentor._manager is None, "a fully discharged manager was kept"


def test_a_CLEAN_removal_still_releases_the_manager():
    """The other direction: retention must be conditional, or every
    deactivation leaks a manager and `is_active()` bookkeeping drifts."""
    tidewall_otel.activate()
    instrumentor = tidewall_otel._instrumentor_instance
    assert instrumentor._manager is not None

    instrumentor._uninstrument()

    assert instrumentor._manager is None
    assert not instrumentor.residuals
