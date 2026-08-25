"""OpenTelemetry instrumentor that activates Tidewall guard enforcement.

This is the main orchestrator. It inherits from OpenTelemetry's
:class:`BaseInstrumentor` and applies ``wrapt`` patches to OpenAI and
Anthropic SDK methods so that every chat-completion call is intercepted
by the Tidewall guard before reaching the underlying provider.

Activation paths (any of these will reach :class:`TidewallInstrumentor`):

1. OTel auto-discovery: ``opentelemetry-instrument python app.py``
2. Programmatic: ``TidewallInstrumentor().instrument()``
3. CLI wrapper: ``tidewall-instrument python app.py``
4. Direct import: ``import tidewall_otel; tidewall_otel.activate()``

The instrumentor degrades gracefully — if neither OpenAI nor Anthropic
SDKs are installed, it logs a warning and applies no patches rather
than raising, so adding ``tidewall-otel`` as a dependency never breaks
an unrelated app.
"""

from __future__ import annotations



import logging
from typing import TYPE_CHECKING, Any, Collection

if TYPE_CHECKING:
    from tidewall_otel._state import State

from tidewall_otel._config import TidewallConfig
from tidewall_otel._guard import TidewallGuard

logger = logging.getLogger("tidewall.otel.instrumentor")

# Module paths used by wrapt to locate the methods to patch. These match
# the module structure of the upstream openai and anthropic SDKs (verified
# against the published packages on PyPI).
_OPENAI_MODULE = "openai.resources.chat.completions.completions"
_ANTHROPIC_MODULE = "anthropic.resources.messages.messages"

# Try to import OpenTelemetry's BaseInstrumentor — if OTel isn't installed,
# substitute a minimal stub so the package still works for the explicit
# `tidewall_otel.activate()` path.
try:
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

    _HAS_OTEL_INSTRUMENTOR = True
except ImportError:
    _HAS_OTEL_INSTRUMENTOR = False

    class BaseInstrumentor:  # type: ignore[no-redef]
        """Minimal stub used when OpenTelemetry isn't installed.

        Mirrors the shape of the real BaseInstrumentor closely enough that
        :class:`TidewallInstrumentor` works the same way regardless of
        whether OTel is present.
        """

        _is_instrumented = False

        def instrumentation_dependencies(self) -> Collection[str]:
            return []

        def instrument(self, **kwargs: Any) -> None:
            if not self._is_instrumented:
                self._instrument(**kwargs)
                self._is_instrumented = True

        def uninstrument(self, **kwargs: Any) -> None:
            if self._is_instrumented:
                self._uninstrument(**kwargs)
                self._is_instrumented = False

        def _instrument(self, **kwargs: Any) -> None:
            raise NotImplementedError

        def _uninstrument(self, **kwargs: Any) -> None:
            raise NotImplementedError


def _installed_version(provider: str) -> str:
    """The installed SDK version, for the manifest's disposition check.

    A surface whose SDK falls outside the tested range is `unverified`, not
    `covered`: the manifest describes what has been verified, and claiming
    coverage for an untested version is a claim with nothing behind it.
    """
    try:
        import importlib.metadata as metadata

        return metadata.version(provider)
    except Exception:
        return "0"


def _sdk_available(module_path: str) -> bool:
    """Return True if a module can be imported without errors."""
    try:
        __import__(module_path)
        return True
    except ImportError:
        return False


class TidewallInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor that adds Tidewall security enforcement.

    Wraps the OpenAI and Anthropic chat-completion methods named in the
    manifest so that:

    1. In an ENFORCING mode (``enforce``/``monitor``), a REPRESENTABLE prompt
       is sent to the guard before reaching the provider. ``dry-run`` skips
       the guard entirely, and a call that cannot be represented losslessly
       is refused in ``enforce`` rather than sent -- the guard is never asked
       about a body it was not shown.
    2. Blocking and transformation decisions are applied inline in
       ``enforce`` mode (see :class:`TidewallConfig`).
    3. ``gen_ai.*`` OpenTelemetry spans are emitted regardless of mode,
       so observability works even when the guard is in monitor mode.

    The unqualified form of (1) -- "every prompt is sent to the guard" -- was
    false in three separate ways while it was written here, and the state
    object exists precisely so the qualifications are reported rather than
    assumed. :attr:`state` is the authority on which surfaces are ``covered``.
    """

    _guard: TidewallGuard | None = None
    _config: TidewallConfig | None = None
    _state: "State | None" = None
    _executor: Any = None
    _manager: Any = None

    @property
    def state(self) -> "State | None":
        """The single State object the installed wrappers write into.

        `activate()` returns THIS object from `state()` rather than building
        its own. A wrapper that downgrades a surface at call time -- say
        `record_unverified` on a construction-time transport escape -- is
        only observable to the caller if both sides hold the same instance.
        Two State objects is how a rewritten provider-bound body coexisted
        with `surfaces={"Completions.create": "covered"}` and `is_active()`
        returning True.
        """
        return self._state

    def instrumentation_dependencies(self) -> Collection[str]:
        """The SDKs this adapter instruments.

        The OTel loader reads this to decide whether to load us at all, so an
        empty list means `opentelemetry-instrument` may skip the adapter
        entirely -- instrumentation silently absent rather than failing.
        """
        return ("openai >= 1.40.0", "anthropic >= 0.27.0")

    def instrument_refusers(self, *, reason: str) -> None:
        """Install a refuser at every manifest boundary.

        Used when activation cannot establish coverage and the operator chose
        `block`: calls fail loudly rather than passing through unchecked while
        the application believes it is protected.
        """
        from tidewall_otel._manager import PatchManager
        from tidewall_otel._manifest import SURFACES
        from tidewall_otel._refuser import make_refuser

        self._manager = PatchManager()
        for surface in SURFACES:
            if not _sdk_available(surface.module):
                continue
            refuser = make_refuser(surface, reason=reason)
            self._manager.install(surface.module, surface.attribute,
                                  lambda w, i, a, k, _r=refuser: _r(w, i, a, k))

        # MARK INSTRUMENTED. uninstrument() is gated on this flag, so
        # bypassing instrument() means deactivate() silently does nothing and
        # the refusers stay installed on the SDK for the rest of the process.
        self._is_instrumented = True
        self._is_instrumented_by_opentelemetry = True


    def _instrument(self, **kwargs: Any) -> None:
        """Install the guard wrappers, transactionally and fully wired.

        EVERY collaborator dispatch needs is constructed here and passed to
        the factories. An earlier version built the components correctly and
        then handed the factories only (guard, config): dispatch dereferenced
        a None executor, the broad handler filed the AttributeError as
        `invariant_violated`, and every enforce call was refused without the
        guard ever being contacted -- while the state reported active. In
        monitor the same fault proceeded UNGUARDED. Nothing caught it because
        every dispatch test called dispatch_sync directly, so they proved the
        component worked and never that activation wires it.

        Installation goes through the PatchManager so a failure part-way
        rolls back: patching four boundaries one at a time can otherwise leave
        some guarded and some not while the agent reports success.
        """
        from tidewall_otel._anthropic_wrapper import (
            make_anthropic_async_wrapper,
            make_anthropic_sync_wrapper,
        )
        from tidewall_otel._execution import BoundedExecutor
        from tidewall_otel._manager import PatchManager
        from tidewall_otel._manifest import SURFACES, disposition_for
        from tidewall_otel._openai_wrapper import (
            make_openai_async_wrapper,
            make_openai_sync_wrapper,
        )
        from tidewall_otel._state import State

        config = kwargs.get("config") or TidewallConfig()
        self._config = config
        self._guard = TidewallGuard(config)
        self._executor = BoundedExecutor()
        self._state = State(lifecycle="installing", mode=config.mode)
        self._hand_off_residual_manager()
        self._manager = PatchManager()

        factories = {
            ("openai", "sync"): make_openai_sync_wrapper,
            ("openai", "async"): make_openai_async_wrapper,
            ("anthropic", "sync"): make_anthropic_sync_wrapper,
            ("anthropic", "async"): make_anthropic_async_wrapper,
        }

        specs = []
        deferred: dict[str, list[str]] = {}
        for surface in SURFACES:
            factory = factories[(surface.provider, surface.kind)]
            wrapper = factory(self._guard, config, self._executor, self._state)

            if not _sdk_available(surface.module):
                # NOT `continue`. The design requires a journal-owned
                # meta-path finder so a module imported AFTER activation is
                # still patched; a surface in a not-yet-imported module is
                # otherwise uncovered forever.
                #
                # `PatchManager` implements this and its unit tests pass, but
                # nothing here called it -- the requirement was built, tested
                # in isolation, and left unreachable. The plan records that v1
                # dropped this same requirement and then reported it covered;
                # leaving the manager's implementation unwired drops it again
                # one layer along, with a green suite over it.
                self._manager.register_surface(surface.module, surface.attribute,
                                               wrapper)
                deferred.setdefault(surface.module, []).append(surface.attribute)
                # DELIBERATELY NOT recorded in `state.surfaces`. `is_active()`
                # is a universal claim over the boundaries PRESENT, and an
                # unimportable module presents none. Recording these would make
                # `is_active()` False for an application that installed only
                # one provider and is fully guarded on it -- the same
                # condemn-everyone over-correction the escape detector had to
                # avoid. The finder still patches them if they ever arrive.
                continue

            specs.append((surface.module, surface.attribute, wrapper))
            self._state.surfaces[surface.attribute] = disposition_for(
                surface, _installed_version(surface.provider)
            )

        # Transactional: any failure rolls back every earlier patch.
        self._manager.install_all(specs)

        if deferred:
            self._manager.install_finder(deferred)

            # TWO CASES, and only one of them is "no boundary present".
            #
            # Finder INSTALLED: the module is absent now and will be patched
            # if it ever arrives, so it contributes no boundary and stays out
            # of `state.surfaces` -- otherwise an app that installed one
            # provider and is fully guarded reports inactive.
            #
            # Finder FAILED: the agent now KNOWS that a boundary which may
            # arrive can never be patched. Staying silent there was the same
            # argument applied where it does not hold, and it let `is_active()`
            # return True while `manager.dispositions` recorded `uncovered`
            # for that very module. The design is explicit: if the finder
            # cannot be installed, in-scope surfaces from not-yet-imported
            # modules are `uncovered`.
            for module, attributes in deferred.items():
                if self._manager.dispositions.get(module) != "uncovered":
                    continue
                for attribute in attributes:
                    # `record_unverified` would be wrong twice here: the
                    # disposition is `uncovered`, not `unverified`, and it
                    # keys `surfaces` by the name given -- passing the module
                    # path would file a module as though it were a surface.
                    self._state.surfaces[attribute] = "uncovered"
                    self._state.record_skip(
                        attribute, reason="finder_not_installed",
                        detail=f"{module} cannot be patched if it is imported later")

            logger.info(
                "Tidewall registered %d boundary(ies) in not-yet-imported "
                "module(s) %s; they are patched on import",
                sum(len(a) for a in deferred.values()), sorted(deferred),
            )

        self._state.lifecycle = "installed"

        # Publish to the package so `tidewall_otel.state()` reflects reality
        # under the OTel entry point, which never goes through the public
        # activate(). Import late: the package imports this module.
        import tidewall_otel

        tidewall_otel._publish(self)

        logger.info(
            "Tidewall instrumentation active (mode=%s, surfaces=%s)",
            config.mode, sorted(self._state.surfaces),
        )

    def _hand_off_residual_manager(self) -> None:
        """Preserve a retained journal before a new manager replaces it.

        The real `BaseInstrumentor` is a SINGLETON, so a second
        instrumentation reaches the same object and overwrites `_manager`.
        `activate()` parks the old one first -- but the documented
        `opentelemetry-instrument` entry point never calls `activate()`: it
        invokes `BaseInstrumentor.instrument()`, which lands here directly.
        Parking only in `activate()` therefore protected the path most under
        test and left the production zero-code path exactly as it was, able
        to orphan a journal whose entries nothing else can ever remove.

        Lives here so BOTH callers get it, rather than being duplicated at
        each entry point where the next one added would forget it.
        """
        previous = getattr(self, "_manager", None)
        if previous is None or not previous.journal:
            return

        self.retry_removal()
        if not previous.journal:
            return

        import tidewall_otel

        if previous not in tidewall_otel._residual_managers:
            tidewall_otel._residual_managers.append(previous)
            logger.warning(
                "Tidewall is re-instrumenting with %d undischarged journal "
                "entry(ies); they are retried on the next deactivate()",
                len(previous.journal),
            )

    def retry_removal(self) -> None:
        """Re-run removal for entries an earlier deactivation could not undo.

        `BaseInstrumentor.uninstrument()` gates on
        `_is_instrumented_by_opentelemetry`, which the first deactivation
        clears -- so the retry the manager supports stayed unreachable
        through the normal path even after `deactivate()` stopped discarding
        the instrumentor. This bypasses that gate deliberately: it is not a
        second uninstrumentation, it is the completion of the first.

        Idempotent. With an empty journal it removes nothing and reports
        nothing.
        """
        self._uninstrument()

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove all applied patches and restore the original SDK methods."""
        # Patches installed through the PatchManager are ITS to remove: it
        # compares before writing and reports not-ours rather than deleting
        # another agent's wrapper. Without this, refusers installed at
        # activation survive deactivation entirely.
        self.residuals: list[str] = []

        manager = getattr(self, "_manager", None)
        if manager is not None:
            for (module_path, attribute), outcome in manager.remove().items():
                if outcome.value != "removed":
                    # STUCK, not removed. Reporting it as uninstrumented would
                    # leave another agent's SDK carrying our wrapper while our
                    # state says we are gone -- the removal-side twin of
                    # reporting enforcement while unguarded.
                    self.residuals.append(
                        f"{module_path}.{attribute}: {outcome.value} -- wrapper NOT removed"
                    )
                    logger.warning("Tidewall could not remove %s.%s: %s",
                                   module_path, attribute, outcome.value)

            # KEEP THE MANAGER while its journal still holds anything. It
            # deliberately retains `not_ours` and `errored` entries so removal
            # can be retried once the foreign wrapper goes or the transient
            # failure clears -- and this line discarded the only object
            # holding their `pre_install_identity`, which made that retry
            # unreachable from the public API.
            #
            # The consequence is not merely untidy. When the foreign layer is
            # later removed, OUR wrapper becomes the live attribute again,
            # with nothing left that can take it off: permanent stale
            # instrumentation on somebody else's SDK. A second deactivate()
            # now retries and can finish the job.
            if not manager.journal:
                self._manager = None
            else:
                logger.warning(
                    "Tidewall is retaining %d journal entry(ies) it could not "
                    "remove; call deactivate() again once the conflicting "
                    "wrapper is gone", len(manager.journal),
                )

        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown()
            self._executor = None

        # NO second removal loop here. There used to be one that walked a
        # parallel `self._patched` list and unwrapped anything that was a
        # `wrapt.FunctionWrapper` with `__wrapped__`. That is a TYPE test, not
        # an ownership test: a wrapper another agent installed AFTER us
        # satisfies it exactly as well as ours does, so deactivation deleted
        # the foreign wrapper and restored the Tidewall layer underneath --
        # the precise inverse of the intent, while the adjacent comment
        # claimed "ONLY unwrap OUR wrapper".
        #
        # PatchManager.remove() compares the CURRENT attribute against the
        # exact object it installed and reports `not_ours` instead of writing.
        # Keeping a second path that cannot make that comparison would mean
        # the guarantee held only when the fallback never ran.
        logger.info("Tidewall instrumentation deactivated")
