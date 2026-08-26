"""The patch manager: transactional installation, and safe removal.

Two defects motivate this module.

PARTIAL INSTALLATION. Patching several boundaries one at a time can
fail halfway, leaving some guarded and some not while the agent reports
success. Installation is therefore transactional: every patch is journalled
with the attribute's pre-install identity, and a failure rolls back in reverse
order.

UNSAFE REMOVAL. Restoring a saved original unconditionally deletes
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
import itertools
import sys
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Iterable


logger = logging.getLogger("tidewall.otel.manager")


#: Sentinel for "no such attribute", distinguishable from a stored None.
_MISSING = object()


class RemovalOutcome(enum.Enum):
    REMOVED = "removed"
    NOT_OURS = "not-ours"
    FOREIGN = "foreign"
    #: `_restore` raised -- the write failed, or something transient went
    #: wrong. Distinct from NOT_OURS, which means we compared and someone else
    #: was there. This one means we never got to compare. RETRYABLE.
    ERRORED = "errored"
    #: The owner object is gone: the class or module we patched no longer
    #: exists, so there is nothing left to restore the original onto. NOT
    #: retryable -- no future call can make a vanished owner reappear, and
    #: retrying forever keeps a journal alive and a lifecycle at `residual`
    #: for the life of the process while achieving nothing. Recorded durably
    #: instead, because silently dropping it is the fail-open this manager
    #: exists to remove.
    UNRECOVERABLE = "unrecoverable"


@dataclass
class JournalEntry:
    kind: str                       # "patch" | "finder"
    module: str = ""
    attribute: str = ""
    pre_install_identity: Any = None
    installed: Any = None
    finder: Any = None
    #: A WEAK reference to the object we patched. Removal must NOT re-resolve
    #: it from `module`/`attribute`: that imports the module if it has left
    #: `sys.modules` and can yield a DIFFERENT instance, whose attribute never
    #: matches `installed` -- so the entry looks foreign forever.
    #:
    #: Weak, not strong: a strong reference would keep the class or module
    #: alive for the life of the process, and it is also what makes
    #: irrecoverability PROVABLE. An attribute can always be recreated, so a
    #: missing one proves nothing; an owner that has been collected can never
    #: carry anything again, and nobody else can reach it either.
    owner_ref: Any = None
    #: Which install transaction created this entry. Identity, not position,
    #: decides what a transaction may undo: Python serialises imports per
    #: module, not across them, so a second provider module importing on
    #: another thread appends into the window between a positional mark and
    #: its slice -- and one module's failure then restores ANOTHER module's
    #: attributes.
    txn: int = 0
    #: Stable identity, assigned once at append. Discharge is by THIS, never
    #: by position: resolving or writing an attribute can run arbitrary
    #: import, descriptor or metaclass code that re-enters the manager and
    #: appends an entry mid-undo. A positional commit computed before that
    #: append deletes the newcomer -- while its wrapper is installed --
    #: making it unremovable. Positions are valid only under a no-reentrancy
    #: invariant these operations do not satisfy.
    seq: int = -1


def _weak(owner: Any) -> Any:
    """A weak reference to `owner`, or the object itself if it cannot take one.

    Some objects (a few C types) do not support weak references. Holding them
    strongly is the safe fallback: it costs a reference for the life of the
    process, and it means `_owner_of` can never report them collected, so
    their entries stay retryable rather than being classified permanent on a
    technicality.
    """
    try:
        return weakref.ref(owner)
    except TypeError:
        return owner


def _owner_of(entry: "JournalEntry") -> Any:
    """The live owner, or None if it has been collected."""
    ref = entry.owner_ref
    if ref is None:
        return None
    return ref() if isinstance(ref, weakref.ref) else ref


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
        #: Entries classified UNRECOVERABLE: the owner no longer carries the
        #: attribute, so no retry can ever discharge them. Kept as a durable
        #: record rather than retried, because retrying achieves nothing and
        #: dropping them silently is the fail-open this manager exists to
        #: remove. Never cleared: a permanent residual is permanent.
        self.permanent_residuals: list[str] = []
        self.dispositions: dict[str, str] = {}
        self._surfaces: dict[str, list[tuple[str, Callable]]] = {}
        self._finder: Any = None
        self._next_seq = 0
        self._txn_ids = itertools.count(1)
        #: Re-entrant: undo resolves and writes attributes, which can run
        #: import, descriptor or metaclass code that re-enters the manager.
        self._lock = threading.RLock()
        #: The transaction this THREAD is inside, if any. Per-thread because
        #: concurrent late imports are the case that broke positional scoping.
        self._active = threading.local()

    def _journal(self, entry: JournalEntry) -> JournalEntry:
        """Stamp an entry with its stable identity and append it.

        One append path, so no entry reaches the journal without a `seq` and
        is silently unmatchable at discharge.
        """
        with self._lock:
            entry.seq = self._next_seq
            self._next_seq += 1
            entry.txn = getattr(self._active, "txn", 0)
            self.journal.append(entry)
        return entry

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

        owner_ref = _weak(owner)

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
            # `owner_ref`, not `owner`. Capturing the owner STRONGLY keeps the
            # patched class or module alive for the life of the process -- and,
            # worse, makes the journal's weak reference immortal, so
            # `UNRECOVERABLE` can never fire and the classification that bounds
            # the residual list has no reachable trigger. The closure needs the
            # owner only as the `instance` argument, which it can dereference at
            # call time; if the owner really has been collected, nothing can be
            # calling this wrapper anyway.
            live_owner = owner_ref() if isinstance(owner_ref, weakref.ref) else owner_ref
            if args and hasattr(original, "__get__") and live_owner is not None:
                receiver, rest = args[0], args[1:]
                bound = original.__get__(receiver, type(receiver))
                return wrapper(bound, receiver, rest, kwargs)
            return wrapper(original, live_owner, args, kwargs)

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
        self._journal(JournalEntry(
            kind="patch", module=module, attribute=attribute,
            pre_install_identity=before, installed=installed,
            owner_ref=owner_ref,
        ))
        return installed

    def install_all(self, specs: Iterable[tuple[str, str, Callable]],
                    fail_on: tuple[str, str] | None = None) -> None:
        """Transactional: any failure rolls back every earlier patch.

        `rollback()` is NON-RAISING by construction (see its docstring), so
        the `raise` below always re-raises the ORIGINAL installation failure.
        A rollback that could raise would REPLACE it: the caller debugs
        "rollback boom" while the actual install failure goes unreported, and
        the journal is never committed -- so state claims nothing is wrong
        while a wrapper stays installed.
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
        transaction = next(self._txn_ids)
        outer = getattr(self._active, "txn", 0)
        self._active.txn = transaction
        try:
            for attribute, wrapper in self._surfaces.get(module, []):
                self.install(module, attribute, wrapper)
        except BaseException:
            # By TRANSACTION, never by position. A slice from a mark taken
            # before the loop contains whatever another thread appended in
            # between, and undoing that restores a module this call never
            # touched -- leaving a live boundary unguarded while the module
            # that installed it reports itself covered.
            added = [entry for entry in self.journal
                     if entry.txn == transaction]
            discharged: set[int] = set()
            try:
                self._undo_entries(added, discharged)
            finally:
                # By seq, over the LIVE journal -- never a slice assignment
                # from the stale `added` snapshot, which would drop anything
                # appended re-entrantly while this rollback ran.
                with self._lock:
                    self.journal[:] = [entry for entry in self.journal
                                       if entry.seq not in discharged]
                # WHAT SURVIVED decides the disposition, not the fact that
                # this call failed. The scoped rollback undoes only THIS
                # call's entries, so an earlier successful install for the
                # same module -- a reload, or a second pass over a module
                # already patched -- is still live and still guarding. Writing
                # "uncovered" unconditionally would report a boundary as
                # unguarded while it is demonstrably patched and working: the
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
        finally:
            self._active.txn = outer

    # -- removal ----------------------------------------------------------

    def _restore(self, entry: JournalEntry) -> RemovalOutcome:
        name = entry.attribute.split(".")[-1]
        owner = _owner_of(entry)

        if owner is None and entry.owner_ref is not None:
            # PROVABLY irrecoverable: the class or module we patched has been
            # collected, so nothing can ever carry the attribute again and
            # nobody else can reach it either.
            return RemovalOutcome.UNRECOVERABLE

        if owner is None:                       # pre-`owner_ref` entry
            owner, name = _resolve_owner(entry.module, entry.attribute)

        # A MISSING attribute is NOT irrecoverable. `hasattr` was used here
        # and is wrong twice over: it invokes descriptor and metaclass code
        # and turns any `AttributeError` into False, so a foreign descriptor
        # that raises on class access while still delegating to our wrapper on
        # instances was classified permanent and stopped being retried -- with
        # the wrapper still live. A module reload makes an attribute briefly
        # absent for the same false positive. Attributes can always be
        # recreated, so absence proves nothing; retaining is the safe answer.
        current = inspect.getattr_static(owner, name, _MISSING)
        if current is _MISSING:
            return RemovalOutcome.ERRORED

        if current is not entry.installed:
            # Something replaced our wrapper. Restoring would delete it.
            return RemovalOutcome.NOT_OURS

        self._write(owner, name, entry.pre_install_identity)
        return RemovalOutcome.REMOVED

    def _undo_entries(self, entries: list[JournalEntry],
                      discharged: set[int]) -> dict[tuple[str, str], RemovalOutcome]:
        """Undo `entries` newest-first, under ONE per-entry failure discipline.

        Every branch -- patch or finder, removal or
        rollback -- contains its own exceptions, records ERRORED, and leaves
        the entry journalled for retry. Four consecutive review rounds each
        Every branch -- patch or finder, removal or rollback -- must contain
        its own exceptions, record ERRORED, and leave the entry journalled for
        retry. One shared loop is the fix for the PATTERN rather than for each
        instance: a new entry kind or a new caller inherits the discipline
        instead of re-implementing it bare, which is how branches end up
        outside the protection their neighbours have.

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
                # Same protection as patches. Bare, a raising
                # `sys.meta_path.remove()` escapes to the caller's `finally`,
                # which erases the journal and `_finder` while the finder is
                # still installed: not retryable, stranded forever.
                try:
                    if entry.finder in sys.meta_path:
                        sys.meta_path.remove(entry.finder)
                except Exception:
                    logger.warning("Tidewall could not remove its import "
                                   "finder from sys.meta_path", exc_info=True)
                    outcomes[(entry.module, entry.attribute)] = RemovalOutcome.ERRORED
                    continue
                discharged.add(entry.seq)
                continue

            # PER-ENTRY. `_restore` can raise -- the module removed from
            # sys.modules, an owner that will not resolve, a write that fails.
            # Letting that escape the loop skips the journal commit, so entries
            # ALREADY restored stay journalled; on retry their attribute no
            # longer holds `entry.installed`, they classify as `not-ours`, and
            # they are retained forever. The retry guarantee would defeat
            # itself on the first exception.
            #
            # ALWAYS COMPARE, INCLUDING ROLLBACK. Restoring
            # `pre_install_identity` unconditionally is tempting on the
            # rollback path -- "these are patches we just made, in a window
            # where nothing else has run" -- but that window is not enforced
            # and does not exist: another thread can replace the attribute, and
            # resolving or writing one can execute arbitrary import, descriptor
            # or metaclass code that patches it. An unconditional rollback
            # silently deletes a wrapper installed after ours, which is the
            # corruption the ownership comparison exists to prevent, committed
            # by the one path that skipped it.
            #
            # So there is one policy and `_restore` is it: transactional
            # rollback must also prove the current value is the object this
            # transaction installed.
            try:
                outcome = self._restore(entry)
            except Exception:
                logger.warning("Tidewall could not remove %s.%s",
                               entry.module, entry.attribute, exc_info=True)
                outcome = RemovalOutcome.ERRORED

            outcomes[(entry.module, entry.attribute)] = outcome

            if outcome is RemovalOutcome.UNRECOVERABLE:
                # Leaves the journal like a success, but for the opposite
                # reason: nothing can be done, so retrying it forever would
                # keep this manager alive and the lifecycle at `residual` for
                # the life of the process while achieving nothing. Recorded
                # first, so stopping the retry never means going quiet.
                record = (f"{entry.module}.{entry.attribute}: unrecoverable -- "
                          f"the class or module that carried it was collected")
                if record not in self.permanent_residuals:
                    self.permanent_residuals.append(record)
                logger.error(
                    "Tidewall cannot restore %s.%s: the class or module that "
                    "carried it has been collected. This is permanent and "
                    "will not be retried.", entry.module, entry.attribute,
                )

            if outcome in (RemovalOutcome.REMOVED, RemovalOutcome.UNRECOVERABLE):
                discharged.add(entry.seq)
        return outcomes

    def _discharge(self) -> dict[tuple[str, str], RemovalOutcome]:
        """Undo the whole journal, then commit whatever survives.

        The commit runs in `finally`: partial progress must not be forgotten,
        or a later retry mistakes our own restored attribute for a foreign
        one. Only entries actually undone leave the journal, so install order
        is preserved and a retry still undoes in reverse. `_finder` survives
        exactly as long as a finder entry does: clearing it while the finder is
        still in `sys.meta_path` would make a stuck finder invisible as well as
        stuck.
        """
        discharged: set[int] = set()
        # Iterate a SNAPSHOT: a late import on another thread appends while
        # this runs, and iterating the live list either raises or silently
        # skips. Anything appended during the undo keeps its entry, exactly
        # as a re-entrant append does.
        with self._lock:
            snapshot = list(self.journal)
        try:
            return self._undo_entries(snapshot, discharged)
        finally:
            # Filter the LIVE journal by seq. Slicing it against positions
            # computed before the undo would delete any entry appended
            # re-entrantly during it -- while that entry's wrapper is
            # installed, so nothing could ever remove it.
            with self._lock:
                self.journal[:] = [entry for entry in self.journal
                                   if entry.seq not in discharged]
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
        return self._discharge()

    def rollback(self) -> None:
        """Reverse order, comparing before every write -- exactly as `remove()`
        does. This path gets no exemption: see `_undo_entries` for why the
        "nothing else has run yet" window does not exist.

        NEVER RAISES (short of a BaseException). `install_all()` calls this
        from its `except` and re-raises the ORIGINAL installation failure; a
        raising rollback would replace that exception AND skip the journal
        commit. An entry that cannot be restored stays journalled -- it is
        still installed, so the journal is telling the truth -- and a later
        `remove()` discharges it.
        """
        self._discharge()

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
        self._journal(JournalEntry(kind="finder", module="sys.meta_path",
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
