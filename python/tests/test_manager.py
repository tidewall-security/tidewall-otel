"""The patch manager (O-6, O-7). Task 6 of the P0 remediation plan."""

import sys
import types

import pytest

from tidewall_otel._manager import PatchManager, RemovalOutcome


class Target:
    """Stands in for an SDK class carrying a patchable method."""

    def create(self, *args, **kwargs):
        return "original"


@pytest.fixture
def module(monkeypatch):
    """A throwaway module holding `Target`, registered in sys.modules so the
    manager can resolve it by name exactly as it resolves a real SDK."""
    mod = types.ModuleType("fake_sdk")
    mod.Target = Target
    pristine = Target.create
    monkeypatch.setitem(sys.modules, "fake_sdk", mod)
    yield mod
    # Restored on teardown: every test here mutates Target.create, and without
    # this the pollution surfaces as a LATER test failing for an unrelated
    # reason -- which is exactly how it presented.
    Target.create = pristine


def wrapper_factory(tag):
    def make(wrapped, instance, args, kwargs):
        return f"{tag}:{wrapped(*args, **kwargs)}"
    return make


# -- install and remove ---------------------------------------------------

def test_install_replaces_the_attribute_and_journals_it(module):
    manager = PatchManager()
    manager.install("fake_sdk", "Target.create", wrapper_factory("tw"))

    assert Target().create() == "tw:original"
    assert len(manager.journal) == 1
    entry = manager.journal[0]
    assert entry.module == "fake_sdk" and entry.attribute == "Target.create"


def test_remove_restores_the_ORIGINAL_not_merely_something(module):
    manager = PatchManager()
    original = Target.create
    manager.install("fake_sdk", "Target.create", wrapper_factory("tw"))
    outcomes = manager.remove()

    assert Target.create is original, "removal did not restore the original"
    assert outcomes == {("fake_sdk", "Target.create"): RemovalOutcome.REMOVED}
    assert Target().create() == "original"


def test_remove_reports_NOT_OURS_when_the_attribute_was_replaced(module):
    """Someone else overwrote our wrapper after we installed it. Restoring
    the original would silently delete their work."""
    manager = PatchManager()
    manager.install("fake_sdk", "Target.create", wrapper_factory("tw"))

    def theirs(self, *a, **k):
        return "foreign"
    Target.create = theirs

    outcomes = manager.remove()
    assert outcomes == {("fake_sdk", "Target.create"): RemovalOutcome.NOT_OURS}
    assert Target.create is theirs, "we removed a wrapper that was not ours"


def test_remove_does_NOT_WRITE_when_the_outcome_is_not_ours(module):
    """Asserted directly rather than inferred from the final value: a write
    followed by a restore leaves the same value and hides the mutation."""
    manager = PatchManager()
    manager.install("fake_sdk", "Target.create", wrapper_factory("tw"))

    def theirs(self, *a, **k):
        return "foreign"
    Target.create = theirs

    writes = []
    original_setattr = type.__setattr__

    def recording_setattr(cls, name, value):
        writes.append((cls, name))
        original_setattr(cls, name, value)

    import unittest.mock as mock
    with mock.patch.object(PatchManager, "_write", side_effect=lambda *a: writes.append(a)):
        manager.remove()
    assert writes == [], "a write occurred on a not-ours outcome"


# -- transactional rollback ----------------------------------------------

def test_a_failure_mid_install_ROLLS_BACK_every_earlier_patch(module):
    """Partial installation is the O-6 defect: some boundaries guarded, some
    not, and the agent reporting success."""
    module.Second = type("Second", (), {"create": lambda self: "second"})
    original_first = Target.create
    original_second = module.Second.create

    manager = PatchManager()

    def exploding(wrapped, instance, args, kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        manager.install_all([
            ("fake_sdk", "Target.create", wrapper_factory("tw")),
            ("fake_sdk", "Second.create", exploding),
        ], fail_on=("fake_sdk", "Second.create"))

    assert Target.create is original_first, "the first patch was not rolled back"
    assert module.Second.create is original_second
    assert manager.journal == [], "the journal survived a rollback"


def test_rollback_restores_in_REVERSE_order(module):
    """Order matters when two patches touch the same attribute: restoring
    forwards leaves the earlier original in place."""
    manager = PatchManager()
    manager.install("fake_sdk", "Target.create", wrapper_factory("first"))
    manager.install("fake_sdk", "Target.create", wrapper_factory("second"))

    assert Target().create() == "second:first:original"
    manager.rollback()
    assert Target().create() == "original"


# -- foreign wrappers -----------------------------------------------------

def test_a_foreign_wrapper_installed_BEFORE_us_is_preserved(module):
    """Tidewall-above: another agent patched first, we patch on top, and
    removal must restore THEIR wrapper rather than the pristine original."""
    def theirs(self, *a, **k):
        return "foreign:original"
    Target.create = theirs

    manager = PatchManager()
    manager.install("fake_sdk", "Target.create", wrapper_factory("tw"))
    manager.remove()

    assert Target.create is theirs


def test_an_interleaved_overwrite_is_RECORDED_as_the_admitted_residual(module):
    """Python has no compare-and-swap for attribute assignment, so a foreign
    wrapper installed between our observation and our write CAN be lost.

    Asserting that it cannot happen would be an unachievable criterion. What
    is achievable, and what this asserts, is that the manager NOTICES and
    records it.
    """
    manager = PatchManager()

    def interleave():
        def theirs(self, *a, **k):
            return "foreign"
        Target.create = theirs
        return theirs

    theirs = manager.install("fake_sdk", "Target.create", wrapper_factory("tw"),
                             _interleave_for_test=interleave)

    assert manager.residuals, "an interleaved overwrite was not recorded"
    assert any("overwrote" in r for r in manager.residuals)


# -- the late-import finder ----------------------------------------------

def test_the_finder_is_journalled_and_removed_with_the_patches(module):
    manager = PatchManager()
    manager.install_finder({"not_yet_imported_sdk"})

    assert any(e.kind == "finder" for e in manager.journal)
    assert any(type(f).__name__ == "_LateImportFinder" for f in sys.meta_path)

    manager.remove()
    assert not any(type(f).__name__ == "_LateImportFinder" for f in sys.meta_path)


def test_a_module_imported_AFTER_activation_is_patched(tmp_path, monkeypatch):
    """The point of the finder: a surface in a not-yet-imported module is
    otherwise uncovered forever."""
    (tmp_path / "late_sdk.py").write_text(
        "class Target:\n    def create(self):\n        return 'original'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "late_sdk", raising=False)

    manager = PatchManager()
    manager.register_surface("late_sdk", "Target.create", wrapper_factory("tw"))
    manager.install_finder({"late_sdk"})

    import late_sdk                                   # noqa: F401  (the point)

    assert late_sdk.Target().create() == "tw:original"
    manager.remove()


def test_importing_twice_patches_ONCE(tmp_path, monkeypatch):
    (tmp_path / "twice_sdk.py").write_text(
        "class Target:\n    def create(self):\n        return 'original'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "twice_sdk", raising=False)

    manager = PatchManager()
    manager.register_surface("twice_sdk", "Target.create", wrapper_factory("tw"))
    manager.install_finder({"twice_sdk"})

    import twice_sdk
    del sys.modules["twice_sdk"]
    import twice_sdk                                  # noqa: F811

    patches = [e for e in manager.journal if e.kind == "patch"]
    assert len(patches) == 1, f"patched {len(patches)} times"
    manager.remove()


def test_a_finder_that_cannot_be_installed_marks_surfaces_UNCOVERED(monkeypatch):
    """Asserted on the manager's own disposition map, not on is_active()."""
    manager = PatchManager()

    class Unwritable(list):
        def insert(self, *args, **kwargs):
            raise RuntimeError("meta_path is locked")

    monkeypatch.setattr(sys, "meta_path", Unwritable(sys.meta_path))
    manager.install_finder({"unreachable_sdk"})

    assert manager.dispositions.get("unreachable_sdk") == "uncovered"


# -- the loader ------------------------------------------------------------

def test_the_loader_patches_AFTER_exec_module(tmp_path, monkeypatch):
    """Patching before exec_module targets attributes that do not exist yet."""
    order = []

    (tmp_path / "order_sdk.py").write_text(
        "class Target:\n    def create(self):\n        return 'original'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "order_sdk", raising=False)

    manager = PatchManager()
    manager.register_surface("order_sdk", "Target.create",
                             lambda w, i, a, k: order.append("patched") or "x")
    original_install = manager.install_for_module

    def recording(fullname):
        order.append("patch_hook")
        return original_install(fullname)

    monkeypatch.setattr(manager, "install_for_module", recording)
    manager.install_finder({"order_sdk"})

    import order_sdk
    assert order_sdk.Target is not None, "the module did not execute"
    assert order == ["patch_hook"], order
    manager.remove()


def test_a_module_whose_exec_module_RAISES_is_never_patched(tmp_path, monkeypatch):
    (tmp_path / "broken_sdk.py").write_text("raise ValueError('bad module')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "broken_sdk", raising=False)

    manager = PatchManager()
    manager.register_surface("broken_sdk", "Target.create", wrapper_factory("tw"))
    manager.install_finder({"broken_sdk"})

    with pytest.raises(ValueError, match="bad module"):
        import broken_sdk                            # noqa: F401

    assert not [e for e in manager.journal if e.kind == "patch"]
    manager.remove()


def test_a_FAILED_install_is_retried_on_the_next_import(tmp_path, monkeypatch):
    """Marking a module patched before the install succeeds means a rollback
    silently skips it on every later import -- permanently."""
    (tmp_path / "retry_sdk.py").write_text(
        "class Target:\n    def create(self):\n        return 'original'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "retry_sdk", raising=False)

    manager = PatchManager()
    manager.register_surface("retry_sdk", "Target.create", wrapper_factory("tw"))
    manager.install_finder({"retry_sdk"})

    attempts = []
    original = manager.install_for_module

    def failing_first(fullname):
        attempts.append(fullname)
        if len(attempts) == 1:
            raise RuntimeError("install failed")
        return original(fullname)

    monkeypatch.setattr(manager, "install_for_module", failing_first)

    with pytest.raises(RuntimeError, match="install failed"):
        import retry_sdk                             # noqa: F401
    # A failed import leaves nothing in sys.modules, so there is nothing to
    # delete -- the retry is simply the next import attempt.
    sys.modules.pop("retry_sdk", None)
    import retry_sdk                                 # noqa: F811

    assert len(attempts) == 2, "a failed install was not retried"
    manager.remove()


def test_the_loader_PRESERVES_create_module(tmp_path, monkeypatch):
    """Returning None changes module-creation semantics for any loader that
    does not use the default."""
    from tidewall_otel._manager import _PatchingLoader

    sentinel = types.ModuleType("sentinel")

    class InnerLoader:
        def create_module(self, spec):
            return sentinel

        def exec_module(self, module):
            pass

        def get_source(self, name):
            return "# source"

    loader = _PatchingLoader(InnerLoader(), "x", lambda name: None)
    assert loader.create_module(None) is sentinel
    assert loader.get_source("x") == "# source", "attribute delegation broken"


def test_remove_restores_STACKED_patches_in_reverse_order(module):
    """Two patches on one attribute: removing forwards restores the earlier
    original first and leaves the later wrapper in place.

    Mutation-testing found `remove()` passing with forward iteration because
    only `rollback()` had a stacked-patch test. Both paths need one.
    """
    manager = PatchManager()
    manager.install("fake_sdk", "Target.create", wrapper_factory("first"))
    manager.install("fake_sdk", "Target.create", wrapper_factory("second"))
    assert Target().create() == "second:first:original"

    manager.remove()
    assert Target().create() == "original", "stacked removal left a wrapper"


def test_identity_capture_is_DESCRIPTOR_SAFE(module):
    """`getattr` on a class triggers the descriptor protocol and can return a
    fresh object each access, so identity comparison against it is
    meaningless. `getattr_static` returns what is actually stored.

    A staticmethod is the cheapest descriptor that demonstrates it: plain
    functions make getattr and getattr_static agree, which is why this
    survived mutation until now.
    """
    import inspect

    class Descriptored:
        @staticmethod
        def create(*args, **kwargs):
            return "original"

    module.Descriptored = Descriptored
    stored = inspect.getattr_static(Descriptored, "create")
    # `getattr` runs the descriptor protocol and hands back a plain function;
    # `getattr_static` returns the staticmethod OBJECT that is actually stored.
    # Restoring the former would put a bare function where a staticmethod was.
    assert isinstance(stored, staticmethod)
    assert not isinstance(Descriptored.create, staticmethod)

    manager = PatchManager()
    manager.install("fake_sdk", "Descriptored.create", wrapper_factory("tw"))
    outcomes = manager.remove()

    assert outcomes[("fake_sdk", "Descriptored.create")] is RemovalOutcome.REMOVED
    assert Descriptored.create() == "original"
    # The stored object is restored, not a bound re-derivation of it.
    assert inspect.getattr_static(Descriptored, "create") is stored


def test_an_unremovable_entry_is_RETAINED_so_removal_can_be_retried(module):
    """Round 7's P1. Clearing the whole journal strands the wrapper forever.

    A patches, then another agent B patches on top.
    `A.remove()` correctly returns `not-ours` -- restoring would delete B --
    but it also cleared A's journal, discarding the `pre_install_identity`
    for the one entry still installed. When B later removes itself, A's
    wrapper is restored to the class as the current value, and A no longer
    holds anything that could remove it.

    So the ownership comparison, which exists to avoid corrupting another
    agent's stack, turned into a permanent leak of our own. Retaining the
    entry makes removal retryable instead.
    """
    manager_a, manager_b = PatchManager(), PatchManager()
    manager_a.install(module.__name__, "Target.create", wrapper_factory("A"))
    manager_b.install(module.__name__, "Target.create", wrapper_factory("B"))

    first = manager_a.remove()
    assert first[(module.__name__, "Target.create")] is RemovalOutcome.NOT_OURS
    assert manager_a.journal, "the entry that could not be removed was discarded"

    manager_b.remove()

    retry = manager_a.remove()
    assert retry[(module.__name__, "Target.create")] is RemovalOutcome.REMOVED
    assert manager_a.journal == [], "a removed entry was retained"
    assert module.Target().create() == "original", (
        f"a wrapper was stranded: {module.Target().create()}")


def test_a_SUCCESSFUL_removal_still_empties_the_journal(module):
    """The other direction: retention must be conditional, or every manager
    accumulates entries it has already undone and a retry re-restores them."""
    manager = PatchManager()
    manager.install(module.__name__, "Target.create", wrapper_factory("only"))

    outcomes = manager.remove()

    assert outcomes[(module.__name__, "Target.create")] is RemovalOutcome.REMOVED
    assert manager.journal == []
    assert module.Target().create() == "original"
