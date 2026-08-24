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

    with tracer.start_as_current_span("gen_ai.chat") as span:
        span.set_attribute(_ATTR_OPERATION, "chat")
        span.set_attribute(_ATTR_SYSTEM, provider)
        span.set_attribute(_ATTR_MODEL, model)

        messages = (guard_input or {}).get("messages")
        if messages:
            try:
                span.set_attribute(
                    _ATTR_INPUT_MESSAGES, json.dumps(messages, default=str)
                )
            except Exception:
                # Serialisation should never break the span — drop silently.
                pass

        yield span


def record_response_in_span(
    span: Any,
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    blocked: bool = False,
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
        if content:
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
