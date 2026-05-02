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

__version__ = "0.1.0"
__all__ = ["activate", "deactivate", "is_active", "TidewallConfig"]

logger = logging.getLogger("tidewall.otel")

_instrumentor_instance = None
_is_active = False


def activate(config: TidewallConfig | None = None) -> None:
    """Activate Tidewall instrumentation for all supported AI SDKs.

    Patches OpenAI and Anthropic SDK chat-completion methods so that every
    call routes through the configured Tidewall guard server before reaching
    the underlying provider. Safe to call multiple times — subsequent calls
    are ignored if instrumentation is already active.

    Args:
        config: Optional explicit configuration. If omitted, configuration
            is loaded from ``TIDEWALL_*`` environment variables.

    Notes:
        In ``dry-run`` mode, no HTTP calls are made to the guard server —
        useful for testing the instrumentation flow without a live backend.
        Configuration errors are logged and instrumentation is silently
        skipped (fail-open) so the host application continues to function.
    """
    global _instrumentor_instance, _is_active

    if _is_active:
        logger.warning("Tidewall instrumentation is already active")
        return

    config = config or TidewallConfig()

    errors = config.validate()
    if errors and config.mode != "dry-run":
        for err in errors:
            logger.error("Tidewall config error: %s", err)
        logger.error("Tidewall instrumentation NOT activated")
        return

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    root_logger = logging.getLogger("tidewall.otel")
    root_logger.setLevel(log_level)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root_logger.addHandler(handler)

    from tidewall_otel._instrumentor import TidewallInstrumentor

    _instrumentor_instance = TidewallInstrumentor()
    _instrumentor_instance.instrument(config=config)
    _is_active = True


def deactivate() -> None:
    """Deactivate Tidewall instrumentation and restore original SDK methods."""
    global _instrumentor_instance, _is_active

    if _instrumentor_instance:
        _instrumentor_instance.uninstrument()
        _instrumentor_instance = None
    _is_active = False


def is_active() -> bool:
    """Return whether Tidewall instrumentation is currently active."""
    return _is_active
