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
import importlib
import importlib.abc
import importlib.machinery
import inspect
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


class RemovalOutcome(enum.Enum):
    REMOVED = "removed"
    NOT_OURS = "not-ours"
    FOREIGN = "foreign"


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
        """Transactional: any failure rolls back every earlier patch."""
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
        for attribute, wrapper in self._surfaces.get(module, []):
            self.install(module, attribute, wrapper)

    # -- removal ----------------------------------------------------------

    def _restore(self, entry: JournalEntry) -> RemovalOutcome:
        owner, name = _resolve_owner(entry.module, entry.attribute)
        current = self._identity(owner, name)

        if current is not entry.installed:
            # Something replaced our wrapper. Restoring would delete it.
            return RemovalOutcome.NOT_OURS

        self._write(owner, name, entry.pre_install_identity)
        return RemovalOutcome.REMOVED

    def remove(self) -> dict[tuple[str, str], RemovalOutcome]:
        """Undo in reverse order, comparing before writing."""
        outcomes: dict[tuple[str, str], RemovalOutcome] = {}
        for entry in reversed(self.journal):
            if entry.kind == "finder":
                if entry.finder in sys.meta_path:
                    sys.meta_path.remove(entry.finder)
                continue
            outcomes[(entry.module, entry.attribute)] = self._restore(entry)
        self.journal.clear()
        self._finder = None
        return outcomes

    def rollback(self) -> None:
        """Reverse order, unconditionally: these are patches WE just made, in
        a window where nothing else has run."""
        for entry in reversed(self.journal):
            if entry.kind == "finder":
                if entry.finder in sys.meta_path:
                    sys.meta_path.remove(entry.finder)
                continue
            owner, name = _resolve_owner(entry.module, entry.attribute)
            self._write(owner, name, entry.pre_install_identity)
        self.journal.clear()

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
        self.journal.append(JournalEntry(kind="finder", finder=finder))
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
