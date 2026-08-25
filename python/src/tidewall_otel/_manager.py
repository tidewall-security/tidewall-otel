"""The patch manager: transactional installation, and safe removal (O-6, O-7).

Two defects motivate this module.

O-6 -- PARTIAL INSTALLATION. Patching several boundaries one at a time can
fail halfway, leaving some guarded and some not while the agent reports
success. Installation is therefore transactional: every patch is journalled
with the attribute's pre-install identity, and a failure rolls back in reverse
order.

O-7 -- UNSAFE REMOVAL. Restoring a saved original unconditionally deletes
whatever replaced our wrapper in the meantime, which may be another agent's
instrumentation. Removal compares first and reports one of three outcomes:
``removed`` when the attribute is still ours, ``not-ours`` when something else
holds it, and ``foreign`` when the pre-install value was itself someone else's
wrapper.

ADMITTED RESIDUAL. Python has no compare-and-swap for attribute assignment, so
a foreign wrapper installed between our observation and our write can still be
lost. That cannot be prevented at this layer; what it can do is notice and
record it, which is what ``residuals`` is for. Asserting the race cannot occur
would be an unachievable criterion.
"""

from __future__ import annotations

import enum
import logging
import importlib
import importlib.abc
import importlib.machinery
import inspect
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable


logger = logging.getLogger("tidewall.otel.manager")


class RemovalOutcome(enum.Enum):
    REMOVED = "removed"
    NOT_OURS = "not-ours"
    FOREIGN = "foreign"
    #: `_restore` raised -- the module went away, the owner would not resolve,
    #: or the write failed. Distinct from NOT_OURS, which means we compared
    #: and someone else was there. This one means we never got to compare.
    ERRORED = "errored"


@dataclass
class JournalEntry:
    kind: str                       # "patch" | "finder"
    module: str = ""
    attribute: str = ""
    pre_install_identity: Any = None
    installed: Any = None
    finder: Any = None


def _resolve_owner(module_name: str, attribute: str) -> tuple[Any, str]:
    """The object holding the attribute, and the final attribute name.

    ``Target.create`` resolves to (the class, "create"); a bare name resolves
    to (the module, name).
    """
    module = sys.modules.get(module_name) or importlib.import_module(module_name)
    parts = attribute.split(".")
    owner: Any = module
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


class PatchManager:
    """Installs wrappers transactionally and removes them safely."""

    def __init__(self) -> None:
        self.journal: list[JournalEntry] = []
        self.residuals: list[str] = []
        self.dispositions: dict[str, str] = {}
        self._surfaces: dict[str, list[tuple[str, Callable]]] = {}
        self._finder: Any = None

    # -- identity ---------------------------------------------------------

    @staticmethod
    def _identity(owner: Any, name: str) -> Any:
        """Descriptor-safe capture.

        ``getattr`` on a class triggers the descriptor protocol and can return
        a fresh bound object each time, so an identity comparison against it
        is meaningless. ``getattr_static`` returns what is actually stored.
        """
        return inspect.getattr_static(owner, name, None)

    @staticmethod
    def _write(owner: Any, name: str, value: Any) -> None:
        setattr(owner, name, value)

    # -- installation -----------------------------------------------------

    def install(self, module: str, attribute: str, wrapper: Callable,
                _interleave_for_test: Callable[[], Any] | None = None) -> Any:
        """Patch one attribute, journalling its pre-install identity."""
        owner, name = _resolve_owner(module, attribute)
        before = self._identity(owner, name)
        original = getattr(owner, name)

        def installed(*args, **kwargs):
            """Adapt the descriptor call to wrapt's wrapper convention.

            wrapt hands a wrapper ``(wrapped, instance, args, kwargs)`` where
            `wrapped` is BOUND and `args` EXCLUDES the receiver. Passing the
            unbound function and args-still-containing-self instead makes
            `signature().bind()` see one positional too many -- which is how
            two components of this codebase disagreed while every unit test
            passed, because the tests called the wrapper directly with
            already-correct arguments.
            """
            if args and hasattr(original, "__get__") and not isinstance(owner, type(None)):
                receiver, rest = args[0], args[1:]
                bound = original.__get__(receiver, type(receiver))
                return wrapper(bound, receiver, rest, kwargs)
            return wrapper(original, owner, args, kwargs)

        installed.__tidewall_wrapper__ = True

        # A hook the interleaving test uses to install a foreign wrapper in
        # the window between observation and write. Production never passes it.
        if _interleave_for_test is not None:
            _interleave_for_test()
            if self._identity(owner, name) is not before:
                self.residuals.append(
                    f"{module}.{attribute}: overwrote a wrapper installed "
                    f"between observation and write (no compare-and-swap exists)"
                )

        self._write(owner, name, installed)
        self.journal.append(JournalEntry(
            kind="patch", module=module, attribute=attribute,
            pre_install_identity=before, installed=installed,
        ))
        return installed

    def install_all(self, specs: Iterable[tuple[str, str, Callable]],
                    fail_on: tuple[str, str] | None = None) -> None:
        """Transactional: any failure rolls back every earlier patch.

        `rollback()` is NON-RAISING by construction (see its docstring), so
        the `raise` below always re-raises the ORIGINAL installation failure.
        When rollback could raise, its exception REPLACED the original here:
        the caller debugged "rollback boom" while the actual install failure
        went unreported -- and the journal was never committed, so state
        claimed nothing was wrong while a wrapper stayed installed.
        """
        try:
            for module, attribute, wrapper in specs:
                if fail_on == (module, attribute):
                    raise RuntimeError("boom")
                self.install(module, attribute, wrapper)
        except BaseException:
            self.rollback()
            raise

    def register_surface(self, module: str, attribute: str, wrapper: Callable) -> None:
        """Record what to patch when ``module`` is imported later."""
        self._surfaces.setdefault(module, []).append((attribute, wrapper))

    def install_for_module(self, module: str) -> None:
        """Transactional per module, exactly like `install_all()`.

        This runs inside `_PatchingLoader.exec_module()`, AFTER the provider
        module has executed. Without the rollback, a failure on a later
        surface raised out of the user's import with an EARLIER surface still
        patched and journalled: a partially guarded module whose disposition
        said nothing at all. Only THIS call's entries are rolled back --
        earlier journal entries are other modules' live coverage (and the
        finder), not ours to undo.
        """
        mark = len(self.journal)
        try:
            for attribute, wrapper in self._surfaces.get(module, []):
                self.install(module, attribute, wrapper)
        except BaseException:
            added = self.journal[mark:]
            discharged: set[int] = set()
            try:
                self._undo_entries(added, discharged, compare=False)
            finally:
                self.journal[mark:] = [entry for index, entry in enumerate(added)
                                       if index not in discharged]
                # WHAT SURVIVED decides the disposition, not the fact that
                # this call failed. The scoped rollback undoes only THIS
                # call's entries, so an earlier successful install for the
                # same module -- a reload, or a second pass over a module
                # already patched -- is still live and still guarding. Writing
                # "uncovered" unconditionally reported a boundary as unguarded
                # while it was demonstrably patched and working, which is the
                # state-lying-about-reality defect this agent exists to avoid,
                # merely pointing the other way.
                #
                # "unverified" rather than "covered" for the partial case: some
                # surfaces of this module took and some did not, so the agent
                # cannot vouch for the module as a whole.
                survives = any(entry.kind == "patch" and entry.module == module
                               for entry in self.journal)
                self.dispositions[module] = "unverified" if survives else "uncovered"
            raise

    # -- removal ----------------------------------------------------------

    def _restore(self, entry: JournalEntry) -> RemovalOutcome:
        owner, name = _resolve_owner(entry.module, entry.attribute)
        current = self._identity(owner, name)

        if current is not entry.installed:
            # Something replaced our wrapper. Restoring would delete it.
            return RemovalOutcome.NOT_OURS

        self._write(owner, name, entry.pre_install_identity)
        return RemovalOutcome.REMOVED

    def _undo_entries(self, entries: list[JournalEntry], discharged: set[int],
                      *, compare: bool) -> dict[tuple[str, str], RemovalOutcome]:
        """Undo `entries` newest-first, under ONE per-entry failure discipline.

        Every branch -- patch or finder, compared removal or unconditional
        rollback -- contains its own exceptions, records ERRORED, and leaves
        the entry journalled for retry. Four consecutive review rounds each
        found one undo branch left outside the protection the previous round
        added to another (round 7: retention; round 12: the patch branch;
        round 13: the finder branch, `rollback`, and `install_for_module`).
        One shared loop is the fix for the PATTERN, not the instance: a new
        entry kind or a new caller inherits the discipline instead of
        re-implementing it bare.

        `discharged` is an out-parameter collecting the indices successfully
        undone, filled AS THE LOOP RUNS, so the caller's `finally` can drop
        exactly those from its journal even when a BaseException (say,
        KeyboardInterrupt) escapes mid-loop. Entries not yet reached are then
        still journalled -- they are still installed, and forgetting them
        strands their wrappers with no `pre_install_identity` left anywhere.
        """
        outcomes: dict[tuple[str, str], RemovalOutcome] = {}
        for index in range(len(entries) - 1, -1, -1):
            entry = entries[index]
            if entry.kind == "finder":
                # Same protection as patches. When this branch was bare, a
                # raising `sys.meta_path.remove()` escaped to the caller's
                # `finally`, which erased the journal and `_finder` while the
                # finder was still installed: not retryable, stranded forever.
                try:
                    if entry.finder in sys.meta_path:
                        sys.meta_path.remove(entry.finder)
                except Exception:
                    logger.warning("Tidewall could not remove its import "
                                   "finder from sys.meta_path", exc_info=True)
                    outcomes[(entry.module, entry.attribute)] = RemovalOutcome.ERRORED
                    continue
                discharged.add(index)
                continue

            # PER-ENTRY. `_restore` can raise -- the module was removed
            # from sys.modules, the owner will not resolve, the write
            # fails. Letting that escape the loop meant the journal
            # commit never ran, so entries ALREADY restored stayed
            # journalled; on retry their attribute no longer held
            # `entry.installed`, they were classified `not-ours`, and they
            # were retained forever. The retry guarantee defeated itself
            # on the first exception.
            try:
                if compare:
                    outcome = self._restore(entry)
                else:
                    # Rollback policy: unconditional. These are patches WE
                    # just made, in a window where nothing else has run, so
                    # there is no foreign wrapper to preserve.
                    owner, name = _resolve_owner(entry.module, entry.attribute)
                    self._write(owner, name, entry.pre_install_identity)
                    outcome = RemovalOutcome.REMOVED
            except Exception:
                logger.warning("Tidewall could not remove %s.%s",
                               entry.module, entry.attribute, exc_info=True)
                outcome = RemovalOutcome.ERRORED

            outcomes[(entry.module, entry.attribute)] = outcome
            if outcome is RemovalOutcome.REMOVED:
                discharged.add(index)
        return outcomes

    def _discharge(self, *, compare: bool) -> dict[tuple[str, str], RemovalOutcome]:
        """Undo the whole journal, then commit whatever survives.

        The commit runs in `finally`: partial progress must not be forgotten,
        or a later retry mistakes our own restored attribute for a foreign
        one. Only entries actually undone leave the journal, so install order
        is preserved and a retry still undoes in reverse. `_finder` survives
        exactly as long as a finder entry does -- clearing it while the
        finder was still in `sys.meta_path` made a stuck finder invisible as
        well as stuck.
        """
        discharged: set[int] = set()
        try:
            return self._undo_entries(self.journal, discharged, compare=compare)
        finally:
            self.journal[:] = [entry for index, entry in enumerate(self.journal)
                               if index not in discharged]
            if not any(entry.kind == "finder" for entry in self.journal):
                self._finder = None

    def remove(self) -> dict[tuple[str, str], RemovalOutcome]:
        """Undo in reverse order, comparing before writing.

        RETAINS entries that could not be removed. Clearing the whole journal
        discarded `pre_install_identity` for exactly the wrappers still
        installed, which strands them permanently:

            A patches, then B patches on top.
            A.remove()  -> not-ours (correct: B is current, restoring would
                           delete B) -- and A forgets everything.
            B.remove()  -> removed, which restores A's wrapper to the class.
            A's wrapper is now current, and nothing on earth can remove it.

        Keeping the entry makes removal RETRYABLE, so a second `remove()`
        after the other agent has gone completes the job. The alternative --
        A deleting B's wrapper to reinstate its own -- is the corruption the
        ownership comparison exists to prevent.
        """
        return self._discharge(compare=True)

    def rollback(self) -> None:
        """Reverse order, unconditionally: these are patches WE just made, in
        a window where nothing else has run.

        NEVER RAISES (short of a BaseException). `install_all()` calls this
        from its `except` and re-raises the ORIGINAL installation failure; a
        raising rollback would replace that exception AND skip the journal
        commit. An entry that cannot be restored stays journalled -- it is
        still installed, so the journal is telling the truth -- and a later
        `remove()` discharges it.
        """
        self._discharge(compare=False)

    # -- late imports -----------------------------------------------------

    def install_finder(self, modules: Iterable[str]) -> None:
        """Insert a meta-path finder so modules imported AFTER activation are
        patched. A surface in a not-yet-imported module is otherwise uncovered
        forever."""
        modules = frozenset(modules)
        finder = _LateImportFinder(self, modules)
        try:
            sys.meta_path.insert(0, finder)
        except Exception:
            # The surfaces this finder would have covered are uncovered, and
            # said so on the manager's own map rather than silently.
            for module in modules:
                self.dispositions[module] = "uncovered"
            return

        self._finder = finder
        # Named, not blank: when removal of the finder fails, the ERRORED
        # outcome is keyed like every patch outcome, so `_uninstrument` can
        # report the stuck finder as a residual instead of reporting clean.
        self.journal.append(JournalEntry(kind="finder", module="sys.meta_path",
                                         attribute="finder", finder=finder))
        for module in modules:
            self.dispositions.setdefault(module, "pending-import")


class _PatchingLoader(importlib.abc.Loader):
    """Delegates to the real loader and patches AFTER the module executes.

    Patching before ``exec_module`` would target attributes that do not exist
    yet. ``create_module`` is delegated rather than replaced with ``None``,
    which would silently change module-creation semantics for any loader not
    using the default.
    """

    def __init__(self, inner: Any, fullname: str, on_loaded: Callable[[str], None]) -> None:
        self._inner = inner
        self._fullname = fullname
        self._on_loaded = on_loaded

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)          # raises => never patched
        self._on_loaded(self._fullname)

    def __getattr__(self, name):
        return getattr(self._inner, name)        # get_source, is_package, ...


class _LateImportFinder(importlib.abc.MetaPathFinder):
    """Patches manifest modules imported after activation.

    Delegation goes back through ``sys.meta_path``, so a naive finder
    re-enters itself; ``_in_progress`` is the recursion guard and ``_patched``
    makes a second import patch once.
    """

    def __init__(self, manager: PatchManager, modules: frozenset[str]) -> None:
        self._manager = manager
        self._modules = modules
        # DEFENSIVE, AND UNREACHABLE AS WRITTEN. Delegating through
        # sys.meta_path would re-enter this finder, but PathFinder.find_spec
        # does not consult sys.meta_path -- verified by probing it. The guard
        # stays because it costs nothing and the delegation target could
        # change, but no test kills its mutation and none pretends to.
        self._in_progress: set[str] = set()
        self._patched: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._modules or fullname in self._in_progress:
            return None                          # let the real finders run
        self._in_progress.add(fullname)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        finally:
            self._in_progress.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, fullname, self._on_loaded)
        return spec

    def _on_loaded(self, fullname: str) -> None:
        if fullname in self._patched:
            return                               # importing twice patches once
        # Marked AFTER success: adding it first means a failed or rolled-back
        # install is silently skipped on every later import, permanently.
        self._manager.install_for_module(fullname)
        self._patched.add(fullname)
        self._manager.dispositions[fullname] = "covered"
