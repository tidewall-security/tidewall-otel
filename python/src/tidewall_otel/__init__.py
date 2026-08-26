"""
Tidewall OTel
=============

Zero-code-change AI security instrumentation for OpenAI and Anthropic SDKs.

Tidewall OTel automatically intercepts AI SDK calls and routes them through
a Tidewall guard server for prompt injection detection, PII redaction,
and policy enforcement. Built on OpenTelemetry's instrumentation framework,
it provides both runtime security AND standard ``gen_ai.*`` observability
spans from a single agent.

Three activation methods are supported:

.. code-block:: bash

    # 1. CLI wrapper (recommended for ops-driven deployments):
    tidewall-instrument python my_app.py

    # 2. OTel auto-instrumentation (zero-code, requires entry-point discovery):
    opentelemetry-instrument python my_app.py

    # 3. Direct import (developer opts in):
    import tidewall_otel
    tidewall_otel.activate()

Configuration is sourced from environment variables (``TIDEWALL_BASE_URL``,
``TIDEWALL_TOKEN`` etc.) or by passing a :class:`TidewallConfig` instance to
:func:`activate`.
"""

from __future__ import annotations

import logging

from tidewall_otel._config import TidewallConfig
from tidewall_otel._manifest import SURFACES
from typing import Any

from tidewall_otel._exceptions import (
    LossyInputError,
    TidewallBlockedError,
    TidewallConfigError,
    TidewallError,
    TidewallRefusedError,
)
from tidewall_otel._refuser import TidewallActivationRefusedError
from tidewall_otel._state import State

__version__ = "0.1.0"
__all__ = [
    "activate",
    "deactivate",
    "is_active",
    "state",
    "TidewallConfig",
    # The exception surface. Documented in the README as
    # `tidewall_otel.TidewallBlockedError`, and reachable at no public name at
    # all until 2026-08-25: an application following the README got an
    # AttributeError, and the only way to catch a block was to import from a
    # PRIVATE module. A library whose primary exception has no public name has
    # no usable error contract.
    "TidewallError",
    "TidewallBlockedError",
    "TidewallRefusedError",
    "LossyInputError",
    "TidewallConfigError",
    "TidewallActivationRefusedError",
]

logger = logging.getLogger("tidewall.otel")

_instrumentor_instance = None
#: MANAGERS whose journal still holds entries removal declined. Parked rather
#: than dropped, so the ONLY route to their `pre_install_identity` survives a
#: re-activation; discharged on any later `deactivate()`.
#:
#: Managers, NOT instrumentors. The real `BaseInstrumentor` is a SINGLETON --
#: `TidewallInstrumentor()` returns the same object every time -- so parking
#: "the previous instrumentor" is a no-op on the production OTel path, and
#: `instrument()` then overwrites `self._manager` and loses the journal just
#: as completely. The manager is the thing that actually holds the entries.
_residual_managers: list = []
#: Wrappers that can NEVER be removed: the class or module that carried them
#: is gone, so there is nothing to restore the original onto. Retrying cannot
#: help, so they are not retried -- but they are not forgotten either, because
#: silently dropping evidence that our code is still installed somewhere is
#: precisely the fail-open this agent exists to remove.
_permanent_residuals: list = []
_state = State()


def activate(config: TidewallConfig | None = None) -> None:
    """Activate Tidewall instrumentation for all supported AI SDKs.

    Patches OpenAI and Anthropic SDK chat-completion methods at the
    boundaries named in the manifest. Safe to call multiple times —
    subsequent calls are ignored if instrumentation is already active.

    What is guaranteed is narrower than "every call is checked", and the
    difference is deliberate:

    * ``enforce`` and ``monitor`` consult the guard before the provider;
      ``dry-run`` never calls the guard at all.
    * A call whose arguments cannot be represented losslessly for the guard
      (``extra_body``, an unmapped shape) is REFUSED in ``enforce`` before
      any guard or provider I/O, rather than checked. The guard is not asked
      about a body it was not shown.
    * A client constructed with its own transport, mounts, middleware or
      request event hooks can rewrite the wire body after inspection. That
      is out of the threat model — anyone able to pass those can equally
      decline to install this agent — but it is DETECTED: the surface is
      downgraded to ``unverified``, :func:`state` records the reason, and
      :func:`is_active` becomes False.

    So the contract is: at the manifest boundaries, in an enforcing mode,
    for representable calls, on a client with no construction-time escape,
    the guard sees the prompt before the provider does. :func:`state` is the
    authority on which of those held; it is not decoration.

    Args:
        config: Optional explicit configuration. If omitted, configuration
            is loaded from ``TIDEWALL_*`` environment variables.

    Notes:
        In ``dry-run`` mode, no HTTP calls are made to the guard server —
        useful for testing the instrumentation flow without a live backend.
        Configuration errors are logged and instrumentation is silently
        skipped (fail-open) so the host application continues to function.
    """
    global _instrumentor_instance, _state

    if _state.lifecycle == "installed":
        logger.warning("Tidewall instrumentation is already active")
        return

    # Policy errors raise from TidewallConfig itself; this catches the
    # connection settings, which the activation-failure policy governs.
    config = config or TidewallConfig()
    errors = config.validate()

    if errors and config.mode != "dry-run":
        _handle_activation_failure(config, errors)
        return

    _configure_logging(config)

    from tidewall_otel._instrumentor import TidewallInstrumentor

    # Parking a retained journal happens in `_instrument()`, not here: the
    # documented `opentelemetry-instrument` entry point never calls this
    # function, so logic placed here protects only the path under test. One
    # place, reached by both callers.

    _instrumentor_instance = TidewallInstrumentor()
    _instrumentor_instance.instrument(config=config)

    # ADOPT the instrumentor's state; do NOT build a second one. The wrappers
    # were handed that object, so downgrades they record (`record_unverified`
    # on a construction-time escape) are only visible to `state()` if it is
    # the SAME object. Two State instances is how a rewritten wire body
    # coexisted with `surfaces={...: "covered"}` and `is_active() is True`.
    #
    # No `or {attribute: "covered"}` fallback: if the instrumentor installed
    # nothing, that is a wiring failure, and fabricating full coverage would
    # be precisely the unevidenced claim this state object exists to prevent.
    _state = _instrumentor_instance.state


def _handle_activation_failure(config: TidewallConfig, errors: list[str]) -> None:
    """Apply ON_ACTIVATION_FAILURE.

    Logging an error and continuing unguarded, while the application believes
    it is protected, is the fail-open this programme exists to remove. Each
    policy is explicit about what the caller gets:

    ``exit``     raise; the process does not continue believing it is guarded
    ``disable``  run on UNGUARDED, with the state saying so
    ``block``    install refusers, so calls fail rather than pass unchecked
    """
    global _state

    detail = "; ".join(errors)
    for error in errors:
        logger.error("Tidewall config error: %s", error)

    if config.on_activation_failure == "exit":
        raise TidewallConfigError(
            f"Tidewall could not activate: {detail}. "
            f"Set TIDEWALL_ON_ACTIVATION_FAILURE=disable to run unguarded, "
            f"or =block to refuse calls instead."
        )

    if config.on_activation_failure == "block":
        _install_refusers(config, reason=detail)
        return

    logger.error("Tidewall NOT active: %s (on_activation_failure=disable)", detail)
    _state = State(lifecycle="uninstalled", mode=config.mode, surfaces={})


def _install_refusers(config: TidewallConfig, reason: str) -> None:
    """Refuse at every boundary rather than pass calls through unchecked."""
    global _instrumentor_instance, _state

    from tidewall_otel._instrumentor import TidewallInstrumentor

    _instrumentor_instance = TidewallInstrumentor()
    _instrumentor_instance.instrument_refusers(reason=reason)
    _state = State(
        lifecycle="installed", mode=config.mode,
        surfaces={surface.attribute: "refusing" for surface in SURFACES},
    )


def _configure_logging(config: TidewallConfig) -> None:
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    root_logger = logging.getLogger("tidewall.otel")
    root_logger.setLevel(log_level)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root_logger.addHandler(handler)


def _publish(instrumentor: Any) -> None:
    """Adopt an instrumentor installed through the OTel entry point.

    `opentelemetry-instrument` never calls :func:`activate`; it constructs the
    instrumentor and calls its hook directly. Without this, the SDK really is
    patched and every call really does reach the guard, while :func:`state`
    still reports the module's initial `uninstalled` -- an operator wiring
    :func:`is_active` into a health check under the documented zero-code
    workflow would read False and conclude they were unprotected.

    Deliberately NOT a call to `activate()`. That direction would recurse:
    activate constructs an instrumentor and calls its hook, which would call
    activate again, and only under the entry point -- never in the direct path
    most tests exercise. Two tests pin both directions of that loop.
    """
    global _instrumentor_instance, _state

    _instrumentor_instance = instrumentor
    _state = instrumentor.state


def state() -> State:
    """The agent's own account of itself.

    Always answerable: an operator asking "is it on?" before activation must
    get an answer rather than an exception or None.
    """
    return _state


def deactivate() -> None:
    """Deactivate Tidewall instrumentation, restoring what can be restored.

    NOT unconditionally ``removed``. `PatchManager.remove()` refuses to write
    when the current attribute is no longer the object it installed -- another
    agent wrapped us afterwards, and deleting their wrapper to reinstate ours
    would corrupt the stack. That refusal is correct, and it leaves a Tidewall
    wrapper live on the SDK underneath theirs.

    Reporting ``removed`` in that case is a fail-open in the reporting layer:
    the instrumentor knows the wrapper survived, and discarding it to publish
    an unconditional ``removed`` state throws the evidence away. An operator
    reading ``removed`` would believe the SDK is pristine while Tidewall code
    still runs on every call for the rest of the process.

    So the lifecycle becomes ``residual`` when anything could not be removed,
    and the reasons are carried on the state as events.
    """
    global _instrumentor_instance, _state

    residuals: list[str] = []

    # Parked residuals first: the conflict that blocked them may have cleared,
    # and each holds the only copy of its entries' pre-install identity.
    for parked in list(_residual_managers):
        for (module_path, attribute), outcome in parked.remove().items():
            if outcome.value not in ("removed", "unrecoverable"):
                residuals.append(
                    f"{module_path}.{attribute}: {outcome.value} -- wrapper NOT removed")
        # Permanent residuals are carried forward once and never retried.
        for record in parked.permanent_residuals:
            if record not in _permanent_residuals:
                _permanent_residuals.append(record)
        if not parked.journal:
            _residual_managers.remove(parked)

    if _instrumentor_instance:
        instrumentor = _instrumentor_instance

        # The FIRST deactivation goes through the OTel gate; a retry cannot,
        # because that gate has already fired. Discarding the instance here --
        # or letting the gate swallow the second call -- left the manager's
        # retained entries unreachable through the public API, which is the
        # only API a caller has. The manager keeping them was necessary and
        # not sufficient.
        # CAPTURE the manager first. `_uninstrument()` clears it once the
        # journal empties, and unrecoverable entries empty the journal -- so
        # reading `permanent_residuals` afterwards found nothing, and the
        # "durable" record lasted exactly one call before a second
        # `deactivate()` reported `removed`. On the real singleton path this
        # is the only manager there is.
        current_manager = getattr(instrumentor, "_manager", None)

        if getattr(instrumentor, "_is_instrumented_by_opentelemetry", False):
            instrumentor.uninstrument()
        else:
            instrumentor.retry_removal()

        if current_manager is not None:
            for record in current_manager.permanent_residuals:
                if record not in _permanent_residuals:
                    _permanent_residuals.append(record)

        # EXTEND. Assigning here threw away every parked residual collected
        # above, so a still-stuck parked manager vanished from the report and
        # lifecycle read `removed` while `_residual_managers` still held a
        # live, retryable entry -- state claiming the SDK is pristine while
        # our wrapper waits under someone else's.
        residuals.extend(getattr(instrumentor, "residuals", ()) or ())

        # Hold the instance only while something is still undischarged, so a
        # later deactivate() can finish once the conflicting wrapper goes.
        manager = getattr(instrumentor, "_manager", None)
        if manager is None or not manager.journal:
            _instrumentor_instance = None

    # Lifecycle from BOTH sources: this deactivation's outcomes AND anything
    # still parked. A clean current removal alongside a stuck parked manager
    # is not `removed`.
    _state = State(
        lifecycle="residual" if (residuals or _residual_managers) else "removed",
        mode=_state.mode,
    )
    for residual in residuals:
        surface, _, reason = residual.partition(": ")
        _state.record_unverified(surface, reason="not_removed", detail=reason)

    # HISTORY, not current state. These name owners that have been collected,
    # so they are not boundaries any more -- and a new class imported under
    # the same module path is a DIFFERENT object, which may well have been
    # patched and removed cleanly. Replaying them as `record_unverified`
    # downgraded that replacement's surface forever, so a reload or plugin
    # system saw every later generation reported unrecoverable on the strength
    # of an earlier one's record.
    for record in _permanent_residuals:
        subject, _, reason = record.partition(": ")
        _state.record_history(subject, reason="unrecoverable", detail=reason)


def is_active() -> bool:
    """Whether the agent is enforcing across every boundary present.

    Delegates to the state's universal claim rather than a separate flag: a
    boolean maintained beside the dimensions can disagree with them, and did.
    """
    return _state.is_active()
