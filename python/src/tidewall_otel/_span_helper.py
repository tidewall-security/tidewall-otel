"""OpenTelemetry span helpers using GenAI semantic conventions.

Creates spans annotated with the ``gen_ai.*`` attributes defined by
OpenTelemetry's GenAI semantic convention working group, so spans
emitted by Tidewall are interoperable with any OTel-compatible
backend (Elastic APM, Tempo, Jaeger, Datadog, etc.).

Gracefully no-ops when OpenTelemetry is not installed: the
:func:`gen_ai_span` context manager yields ``None`` and
:func:`record_response_in_span` becomes a no-op. This keeps
``tidewall-otel`` usable even in environments where OTel is
intentionally not installed (e.g. quick smoke tests).
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger("tidewall.otel.span")

# Import OpenTelemetry conditionally so the package still loads when OTel
# isn't installed. The boolean below is consulted by every helper before
# touching any OTel API.
try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

# Attribute names per OTel GenAI semantic conventions (v1.x).
# https://opentelemetry.io/docs/specs/semconv/gen-ai/
_ATTR_OPERATION = "gen_ai.operation.name"
_ATTR_SYSTEM = "gen_ai.system"
_ATTR_MODEL = "gen_ai.request.model"
_ATTR_INPUT_MESSAGES = "gen_ai.input.messages"
_ATTR_OUTPUT_MESSAGES = "gen_ai.output.messages"
_ATTR_FINISH_REASONS = "gen_ai.response.finish_reasons"

_TRACER_NAME = "tidewall.otel"
_TRACER_VERSION = "0.1.0"


def _get_tracer() -> Any:
    """Return an OpenTelemetry tracer, or ``None`` if OTel isn't installed."""
    if not _HAS_OTEL:
        return None
    return trace.get_tracer(_TRACER_NAME, _TRACER_VERSION)


@contextmanager
def gen_ai_span(
    *,
    provider: str,
    model: str,
    guard_input: dict | None = None,
    include_input: bool = False,
) -> Generator[Any, None, None]:
    """Open a ``gen_ai.chat`` span with the standard request attributes.

    Args:
        provider: ``gen_ai.system`` value — ``"openai"``, ``"anthropic"``, etc.
        model: Model identifier, recorded as ``gen_ai.request.model``.
        guard_input: The complete guard input; its ``messages`` are
            serialized into ``gen_ai.input.messages`` if span content is
            enabled.

    Yields:
        The active span, or ``None`` if OTel isn't installed.

    Note:
        Recording prompt content on spans can include sensitive data
        depending on application policy. Most users export to a backend
        that supports redaction or sampling. Consider toggling this off
        in environments handling regulated data.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return

    # THE WHOLE LIFECYCLE IS BEST-EFFORT. Every dispatch opens a span before
    # contacting the guard, which makes the tracing pipeline a runtime
    # dependency of the security path: unguarded, a processor raising in
    # `start_as_current_span` or `on_start` prevents the guarded call
    # entirely, and one raising on `on_end` replaces the application's own
    # exception while unwinding. Observability breaking the thing it observes
    # is the failure mode this agent exists to avoid, arriving from the
    # telemetry side.
    try:
        manager = tracer.start_as_current_span(
            "gen_ai.chat", record_exception=False, set_status_on_exception=False
        )
        span = manager.__enter__()
    except Exception:
        logger.debug("could not open a gen_ai span", exc_info=True)
        yield None
        return

    try:
        try:
            span.set_attribute(_ATTR_OPERATION, "chat")
            span.set_attribute(_ATTR_SYSTEM, provider)
            span.set_attribute(_ATTR_MODEL, model)

            # `include_input` DEFAULTS TO FALSE and the caller passes the
            # surface's own `span_input` flag. Serialising the conversation
            # unconditionally is the P0 this programme opened with: prompts
            # carry credentials, customer data and system prompts, and a span
            # exporter ships them to an observability backend. The manifest
            # declares the policy per surface; this function must not decide
            # it.
            messages = (guard_input or {}).get("messages") if include_input else None
            if messages:
                span.set_attribute(
                    _ATTR_INPUT_MESSAGES, json.dumps(messages, default=str)
                )
        except Exception:
            logger.debug("could not annotate a gen_ai span", exc_info=True)

        try:
            yield span
        except BaseException as exc:
            # TYPE ONLY -- never `str(exc)`, which is the leak itself.
            try:
                span.set_attribute("tidewall.error.type", type(exc).__name__)
                span.set_status(StatusCode.ERROR)
            except Exception:
                logger.debug("could not record an error on a span", exc_info=True)
            raise
    finally:
        # Closing must not replace the caller's exception or its result.
        # `None, None, None` deliberately: exception recording is disabled on
        # this span and the status has already been set above, so there is
        # nothing for OTel to add and something for it to leak.
        try:
            manager.__exit__(None, None, None)
        except Exception:
            logger.debug("could not close a gen_ai span", exc_info=True)


def record_response_in_span(
    span: Any,
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    blocked: bool = False,
    include_output: bool = False,
) -> None:
    """Annotate a ``gen_ai.chat`` span with response details.

    Safely a no-op when OTel isn't installed or the supplied span is None.
    When ``blocked`` is True, the span status is set to ERROR and the
    finish reason is set to ``tidewall_content_filter`` so dashboards
    can distinguish guard blocks from upstream provider errors.
    """
    if span is None or not _HAS_OTEL:
        return

    try:
        # Same rule as the input side: `include_output` defaults to False and
        # the caller passes the surface's `span_output` flag. Completions carry
        # exactly the material prompts do.
        if content and include_output:
            span.set_attribute(
                _ATTR_OUTPUT_MESSAGES,
                json.dumps([{"role": "assistant", "content": content}]),
            )
        if finish_reason:
            span.set_attribute(_ATTR_FINISH_REASONS, [finish_reason])
        if blocked:
            span.set_attribute(_ATTR_FINISH_REASONS, ["tidewall_content_filter"])
            span.set_status(StatusCode.ERROR, "Blocked by Tidewall policy")
    except Exception:
        logger.debug("Failed to record response in span", exc_info=True)
