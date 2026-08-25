"""The patch manager (O-6, O-7). Task 6 of the P0 remediation plan."""

import inspect
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

    # Derived from the specs, so adding a third boundary to the install list
    # extends the assertion instead of silently escaping it.
    for owner, attribute, expected in (
        (Target, "create", original_first),
        (module.Second, "create", original_second),
    ):
        assert getattr(owner, attribute) is expected, (
            f"{owner.__name__}.{attribute} was not rolled back")
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


def test_a_RAISING_restore_does_not_strand_the_entries_already_removed(monkeypatch):
    """Round 12's P1: the retry guarantee defeated itself on the first error.

    `remove()` restores in reverse order and only committed the reduced
    journal after the whole loop succeeded. If a later entry restored cleanly
    and an earlier one raised -- its module gone from `sys.modules`, say --
    the assignment never ran and the original journal survived, entries
    already restored included. On retry those attributes no longer held
    `entry.installed`, so they were classified `not-ours` and retained
    forever: a journal entry no retry could ever discharge.

    Also exercises the handler itself. `logger` was referenced there and
    never defined in this module, so the first exception would have raised
    NameError out of the error path -- invisible because nothing reached it.
    """
    import sys
    import types

    modules = {}
    for name in ("r12_first", "r12_second"):
        module = types.ModuleType(name)
        module.Target = type("Target", (), {"create": staticmethod(lambda: "original")})
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module

    manager = PatchManager()
    manager.install("r12_first", "Target.create", wrapper_factory("first"))
    manager.install("r12_second", "Target.create", wrapper_factory("second"))

    # Make the FIRST-installed entry's write fail. Removing its module from
    # sys.modules no longer does that: entries hold their owner object
    # directly, so a vanished module is not a failure any more -- which is the
    # point of holding it. A raising write is now the honest construction.
    real_write = manager._write

    def failing_write(owner, name, value):
        if owner is modules["r12_first"].Target:
            raise RuntimeError("write boom")
        return real_write(owner, name, value)

    manager._write = failing_write

    outcomes = manager.remove()

    assert outcomes[("r12_second", "Target.create")] is RemovalOutcome.REMOVED
    assert outcomes[("r12_first", "Target.create")] is RemovalOutcome.ERRORED
    assert modules["r12_second"].Target.create() == "original"

    retained = {(entry.module, entry.attribute) for entry in manager.journal}
    assert retained == {("r12_first", "Target.create")}, (
        f"a successfully restored entry was stranded in the journal: {retained}")

    # And the retry discharges it once the module is back.
    manager._write = real_write
    retry = manager.remove()

    assert retry[("r12_first", "Target.create")] is RemovalOutcome.REMOVED
    assert manager.journal == [], "the journal was never discharged"
    assert modules["r12_first"].Target.create() == "original"


def test_a_finder_that_cannot_be_REMOVED_is_retained_for_retry(monkeypatch):
    """Round 13's finding 1: the round-12 protection wrapped the patch branch
    and left the finder branch bare.

    A raising `sys.meta_path.remove()` escaped to the `finally`, whose
    `retained` list never contains the finder -- so the journal and `_finder`
    were erased while the finder was still installed. Unlike a retained patch
    entry, nothing could ever retry it: stranded permanently.
    """
    # Sandbox meta_path FIRST, so the finder never enters the real one and
    # teardown restores a clean interpreter.
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))

    manager = PatchManager()
    manager.install_finder({"stuck_sdk"})
    finder = manager._finder

    class Unremovable(list):
        def remove(self, *args, **kwargs):
            raise RuntimeError("meta_path is locked")

    monkeypatch.setattr(sys, "meta_path", Unremovable(sys.meta_path))
    outcomes = manager.remove()

    # Reported, not raised -- and reported under a key `_uninstrument` can
    # turn into a residual instead of claiming a clean removal.
    assert outcomes[("sys.meta_path", "finder")] is RemovalOutcome.ERRORED
    assert any(e.kind == "finder" for e in manager.journal), (
        "the stuck finder was forgotten while still installed")
    assert manager._finder is finder, "the reference to the stuck finder was erased"
    assert finder in sys.meta_path

    # And the retry discharges it once meta_path is writable again.
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    manager.remove()

    assert finder not in sys.meta_path
    assert manager.journal == [], "the discharged finder was retained"
    assert manager._finder is None


def test_a_FAILING_rollback_does_not_mask_the_original_install_error(module):
    """Round 13's finding 2: `install_all()` called `rollback()` unprotected.

    A rollback exception REPLACED the installation exception -- the caller
    debugged "rollback boom" while the actual install failure went unreported
    -- and the journal was never committed, so an installed wrapper survived
    an `install_all()` that claimed, by raising, to have installed nothing.
    """
    manager = PatchManager()
    manager.install("fake_sdk", "Target.create", wrapper_factory("tw"))

    import unittest.mock as mock
    with mock.patch.object(PatchManager, "_write",
                           side_effect=RuntimeError("rollback boom")):
        # `Target.missing` fails BEFORE any write, so within this block the
        # only `_write` calls are rollback's.
        with pytest.raises(AttributeError, match="missing"):
            manager.install_all([("fake_sdk", "Target.missing",
                                  wrapper_factory("tw"))])

    # The journal tells the truth: the entry rollback could not restore is
    # still installed, and stays journalled rather than being cleared.
    assert [(e.module, e.attribute) for e in manager.journal] == [
        ("fake_sdk", "Target.create")]
    assert Target().create() == "tw:original"

    # And a later remove() discharges it once writes work again.
    outcomes = manager.remove()
    assert outcomes[("fake_sdk", "Target.create")] is RemovalOutcome.REMOVED
    assert Target().create() == "original"


def test_a_failure_mid_LATE_install_rolls_back_ONLY_that_modules_surfaces(
        module, monkeypatch):
    """Round 13's finding 3: `install_for_module()` was not transactional.

    It runs inside `exec_module()` on import. A failure on a later surface
    raised out of the user's import with an earlier surface still patched and
    journalled, and the module got no disposition at all. The rollback must
    also be SCOPED: entries journalled before this import are other modules'
    live coverage, not part of the failed transaction.
    """
    other = types.ModuleType("other_sdk")
    other.Target = type("Target", (), {"create": lambda self: "other"})
    monkeypatch.setitem(sys.modules, "other_sdk", other)

    manager = PatchManager()
    manager.install("other_sdk", "Target.create", wrapper_factory("pre"))

    manager.register_surface("fake_sdk", "Target.create", wrapper_factory("tw"))
    manager.register_surface("fake_sdk", "Missing.create", wrapper_factory("tw"))

    original = Target.create
    with pytest.raises(AttributeError, match="Missing"):
        manager.install_for_module("fake_sdk")

    # This module's partial patch is rolled back and un-journalled...
    assert Target.create is original, "an earlier surface stayed patched"
    assert [(e.module, e.attribute) for e in manager.journal] == [
        ("other_sdk", "Target.create")], "the rollback was not scoped"
    # ...coverage installed before the failed import is untouched...
    assert other.Target().create() == "pre:other"
    # ...and the module is dispositioned, not silent, until a retry covers it.
    assert manager.dispositions.get("fake_sdk") == "uncovered"

    manager.remove()


def test_a_failed_REINSTALL_does_not_report_a_live_patch_as_uncovered(monkeypatch):
    """Round 14's finding, recovered from a killed reviewer's log.

    `install_for_module` rolls back only ITS call's entries -- earlier ones
    are live coverage, not part of this transaction -- but then wrote
    `uncovered` unconditionally. So after a reload, or any second pass over a
    module already patched, the disposition said the boundary was unguarded
    while it was demonstrably patched and still guarding.

    That is the same defect this agent exists to prevent, pointing the other
    way: state disagreeing with reality. Under-claiming is the safer
    direction, but an operator reading `uncovered` for a guarded module still
    acts on a false report.

    `unverified` rather than `covered` for the partial case: some surfaces of
    the module took and some did not, so the agent cannot vouch for the module
    as a whole.
    """
    import sys
    import types

    module = types.ModuleType("reinstall_sdk")
    module.Target = type("Target", (), {"create": staticmethod(lambda: "original")})
    monkeypatch.setitem(sys.modules, "reinstall_sdk", module)

    manager = PatchManager()
    manager.register_surface("reinstall_sdk", "Target.create", wrapper_factory("first"))
    manager.install_for_module("reinstall_sdk")

    assert module.Target.create() == "first:original", "precondition: it patched"

    # A second pass adds a surface that cannot resolve.
    manager.register_surface("reinstall_sdk", "Target.absent", wrapper_factory("second"))
    with pytest.raises(AttributeError):
        manager.install_for_module("reinstall_sdk")

    assert module.Target.create() == "first:original", (
        "the scoped rollback undid an earlier call's live patch")
    assert manager.dispositions["reinstall_sdk"] == "unverified", (
        f"a live patch was reported as {manager.dispositions['reinstall_sdk']!r}")


def test_a_failed_FIRST_install_is_still_reported_uncovered(monkeypatch):
    """The other direction. If nothing of this module survived, `uncovered`
    is the truthful answer and must not be softened to `unverified`."""
    import sys
    import types

    module = types.ModuleType("firstfail_sdk")
    module.Target = type("Target", (), {"create": staticmethod(lambda: "original")})
    monkeypatch.setitem(sys.modules, "firstfail_sdk", module)

    manager = PatchManager()
    manager.register_surface("firstfail_sdk", "Target.absent", wrapper_factory("x"))
    with pytest.raises(AttributeError):
        manager.install_for_module("firstfail_sdk")

    assert manager.dispositions["firstfail_sdk"] == "uncovered"
    assert module.Target.create() == "original"


def test_ROLLBACK_refuses_to_delete_a_wrapper_installed_after_ours(module):
    """Adversarial review finding 1. Rollback compared nothing before writing.

    Its justification was that these are patches WE just made, in a window
    where nothing else has run. That window is not enforced and does not
    exist: another thread can replace the attribute, and resolving or writing
    one can execute arbitrary import, descriptor or metaclass code that
    patches it. Rollback then silently deleted a wrapper installed after ours
    -- the exact corruption the ownership comparison exists to prevent,
    committed by the one path that skipped it.
    """
    import wrapt

    manager = PatchManager()
    manager.install(module.__name__, "Target.create", wrapper_factory("tw"))

    # Someone else wraps our surface before the transaction unwinds.
    wrapt.wrap_function_wrapper(module.__name__, "Target.create",
                                lambda w, i, a, k: "foreign")
    theirs = inspect.getattr_static(module.Target, "create")

    manager.rollback()

    assert inspect.getattr_static(module.Target, "create") is theirs, (
        "rollback deleted a wrapper installed after ours")
    assert manager.journal, "the entry it could not roll back was discarded"


def test_an_entry_appended_RE_ENTRANTLY_during_undo_survives_the_commit(module):
    """Adversarial review finding 2, and why positions were unsound.

    Resolving or writing a patched attribute can execute arbitrary import,
    descriptor or metaclass code, which can re-enter this manager and append
    a journal entry mid-undo. The commit filtered by POSITION, computed
    before that append -- so the newcomer was deleted from the journal while
    its wrapper was installed, and nothing could ever remove it.

    Discharge is by stable `seq` over the LIVE journal, so an entry that did
    not exist when the undo began cannot be discharged by it.
    """
    manager = PatchManager()
    manager.install(module.__name__, "Target.create", wrapper_factory("first"))

    newcomer = {}
    real_restore = manager._restore

    def restore_then_reenter(entry):
        outcome = real_restore(entry)
        if not newcomer:            # exactly what user code inside setattr does
            module.Second = type("Second", (), {"create": lambda self: "second"})
            manager.install(module.__name__, "Second.create",
                            wrapper_factory("late"))
            newcomer["seq"] = manager.journal[-1].seq
        return outcome

    manager._restore = restore_then_reenter
    manager.remove()

    assert newcomer, "the re-entrant install never ran"
    assert newcomer["seq"] in {entry.seq for entry in manager.journal}, (
        "an entry appended during the undo was deleted from the journal "
        "while its wrapper was still installed")
    assert getattr(inspect.getattr_static(module.Second, "create"),
                   "__tidewall_wrapper__", False), (
        "precondition: the re-entrant install really did patch")

    # Still removable, which is the whole point.
    outcomes = manager.remove()
    assert outcomes[(module.__name__, "Second.create")] is RemovalOutcome.REMOVED
    assert manager.journal == []


def test_a_COLLECTED_owner_is_unrecoverable_and_recorded():
    """The only PROVABLE irrecoverable condition: the owner is gone.

    An attribute can always be recreated, so a missing one proves nothing --
    an earlier version used `hasattr` and classified a live, retryable wrapper
    as permanent whenever a foreign descriptor raised on class access or a
    reload made the attribute briefly absent. An owner that has been collected
    can never carry anything again, and nobody else can reach it either.

    This drives a REAL collection rather than a manufactured dead reference.
    It only became reachable once the installed wrapper stopped capturing its
    owner strongly: that capture kept the patched class alive for the life of
    the process and made the journal's weak reference immortal, so this
    classification had no trigger at all.
    """
    import gc
    import sys
    import types
    import weakref

    victim = types.ModuleType("collected_sdk")
    victim.Target = type("Target", (), {"create": staticmethod(lambda: "orig")})
    sys.modules["collected_sdk"] = victim

    manager = PatchManager()
    manager.install("collected_sdk", "Target.create", wrapper_factory("tw"))
    owner_watch = weakref.ref(victim.Target)

    del victim.Target
    del sys.modules["collected_sdk"]
    del victim
    gc.collect()

    assert owner_watch() is None, (
        "the owner is still referenced, so this test proves nothing")

    outcomes = manager.remove()

    assert outcomes[("collected_sdk", "Target.create")] is RemovalOutcome.UNRECOVERABLE
    assert manager.journal == [], "an entry nothing can discharge was kept"
    assert manager.permanent_residuals, "it was dropped without a record"
    assert "collected" in manager.permanent_residuals[0]

    again = manager.remove()
    assert again == {}
    assert len(manager.permanent_residuals) == 1


def test_the_installed_wrapper_does_not_PIN_the_class_it_patches():
    """The capture that made the classification unreachable, asserted directly.

    A strong capture kept every patched class or module alive for the life of
    the process. It also made the journal's weak reference immortal, which is
    why `UNRECOVERABLE` could never fire.
    """
    import gc
    import sys
    import types
    import weakref

    victim = types.ModuleType("pinned_sdk")
    victim.Target = type("Target", (), {"create": staticmethod(lambda: "orig")})
    sys.modules["pinned_sdk"] = victim

    manager = PatchManager()
    manager.install("pinned_sdk", "Target.create", wrapper_factory("tw"))
    watch = weakref.ref(victim.Target)

    del victim.Target
    del sys.modules["pinned_sdk"]
    del victim
    gc.collect()

    assert watch() is None, (
        "the installed wrapper is pinning the class it patched")


def test_a_MISSING_attribute_is_retryable_not_permanent(module):
    """`hasattr` was wrong twice: it runs descriptor and metaclass code, and
    turns any `AttributeError` into False. A foreign descriptor that raises on
    class access while still delegating to our wrapper on instances was
    classified permanent and stopped being retried -- with the wrapper live.
    """
    manager = PatchManager()
    manager.install(module.__name__, "Target.create", wrapper_factory("tw"))

    del module.Target.create            # absent now; recreatable later

    outcomes = manager.remove()

    assert outcomes[(module.__name__, "Target.create")] is not RemovalOutcome.UNRECOVERABLE
    assert manager.journal, "a recreatable attribute was discharged permanently"
    assert manager.permanent_residuals == []


def test_a_NOT_OURS_entry_is_still_retained_for_retry(module):
    """The other direction, and the one that must not be swept up.

    A foreign wrapper can be removed later, so `not_ours` stays retryable.
    Only a vanished owner is permanent.
    """
    import wrapt

    manager = PatchManager()
    manager.install(module.__name__, "Target.create", wrapper_factory("tw"))
    wrapt.wrap_function_wrapper(module.__name__, "Target.create",
                                lambda w, i, a, k: "foreign")

    outcomes = manager.remove()

    assert outcomes[(module.__name__, "Target.create")] is RemovalOutcome.NOT_OURS
    assert manager.journal, "a retryable entry was discarded"
    assert manager.permanent_residuals == [], (
        "a removable conflict was recorded as permanent")


def test_a_vanished_MODULE_no_longer_breaks_removal_at_all(module, monkeypatch):
    """Entries hold their owner object, so removal never re-imports.

    Re-resolving by name imported the module if it had left `sys.modules`,
    which can yield a DIFFERENT module instance whose attribute can never
    match `installed` -- so the entry looked foreign forever. Holding the
    owner removes the re-import and the ambiguity together.
    """
    import sys

    manager = PatchManager()
    manager.install(module.__name__, "Target.create", wrapper_factory("tw"))

    monkeypatch.delitem(sys.modules, module.__name__)

    outcomes = manager.remove()

    assert outcomes[(module.__name__, "Target.create")] is RemovalOutcome.REMOVED
    assert module.Target().create() == "original"
