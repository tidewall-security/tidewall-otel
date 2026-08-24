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
from tidewall_otel._exceptions import TidewallConfigError
from tidewall_otel._manifest import SURFACES
from typing import Any

from tidewall_otel._state import State

__version__ = "0.1.0"
__all__ = ["activate", "deactivate", "is_active", "TidewallConfig"]

logger = logging.getLogger("tidewall.otel")

_instrumentor_instance = None
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
    if _instrumentor_instance:
        _instrumentor_instance.uninstrument()
        residuals = list(getattr(_instrumentor_instance, "residuals", ()) or ())
        _instrumentor_instance = None

    _state = State(lifecycle="residual" if residuals else "removed",
                   mode=_state.mode)
    for residual in residuals:
        surface, _, reason = residual.partition(": ")
        _state.record_unverified(surface, reason="not_removed", detail=reason)


def is_active() -> bool:
    """Whether the agent is enforcing across every boundary present.

    Delegates to the state's universal claim rather than a separate flag: a
    boolean maintained beside the dimensions can disagree with them, and did.
    """
    return _state.is_active()
