"""OpenAI SDK ``wrapt`` wrappers for Tidewall guard integration.

Wraps :py:meth:`openai.resources.chat.completions.completions.Completions.create`
(and the async variant) to inject inline Tidewall guard calls before the
underlying request is dispatched to the OpenAI API.

Each wrapper:

1. Normalizes the OpenAI message list into the format the guard expects.
2. Calls the Tidewall guard for ``event_type="input"``.
3. In ``enforce`` mode: raises :class:`TidewallBlockedError` on block, or
   replaces ``messages`` with the redacted version on transform.
4. Opens a ``gen_ai.chat`` OTel span (capturing input messages) and runs
   the original SDK call inside the span context.
5. Records the response on the span when it is non-streaming.
"""

from __future__ import annotations

import logging
from typing import Any

from tidewall_otel._config import TidewallConfig
from tidewall_otel._exceptions import TidewallBlockedError
from tidewall_otel._guard import TidewallGuard
from tidewall_otel._manifest import OPENAI_CHAT_ASYNC, OPENAI_CHAT_SYNC
from tidewall_otel._normalizer import normalize
from tidewall_otel._span_helper import gen_ai_span, record_response_in_span

logger = logging.getLogger("tidewall.otel.openai")


def make_openai_sync_wrapper(
    guard: TidewallGuard, config: TidewallConfig
) -> Any:
    """Build a wrapt-compatible sync wrapper for ``Completions.create``.

    Returns a wrapper closure that captures the supplied ``guard`` and
    ``config`` instances. ``wrapt`` will call this closure with the
    wrapped function, instance, args and kwargs at runtime.
    """

    def wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        messages = list(kwargs.get("messages") or [])
        model = str(kwargs.get("model", "unknown"))
        is_stream = bool(kwargs.get("stream", False))

        # --- INPUT GUARD ---
        normalized = normalize(OPENAI_CHAT_SYNC, kwargs)
        input_result = guard.check(
            guard_input=normalized,
            event_type="input",
            model=model,
            llm_provider="openai",
        )

        if input_result and config.mode == "enforce":
            if input_result.blocked:
                raise TidewallBlockedError(
                    input_result.summary, input_result.detectors
                )
            if input_result.transformed and input_result.guard_output:
                kwargs["messages"] = input_result.guard_output["messages"]
                logger.info(
                    "Input transformed by Tidewall (sensitive data redacted)"
                )

        # --- OTel SPAN + ORIGINAL CALL ---
        with gen_ai_span(provider="openai", model=model, guard_input=normalized) as span:
            response = wrapped(*args, **kwargs)

            if not is_stream:
                try:
                    record_response_in_span(
                        span, content=response.choices[0].message.content
                    )
                except (IndexError, AttributeError, KeyError):
                    # Streaming or non-standard response shape — silently
                    # skip recording; observability is best-effort here.
                    pass

        return response

    return wrapper


def make_openai_async_wrapper(
    guard: TidewallGuard, config: TidewallConfig
) -> Any:
    """Build a wrapt-compatible async wrapper for ``AsyncCompletions.create``.

    Functionally identical to :func:`make_openai_sync_wrapper` but uses
    ``await`` for the underlying SDK call. The guard call itself is
    synchronous — the latency is dominated by the AI provider, so a
    synchronous guard call is acceptable for the POC. A future revision
    may switch to an async HTTP client.
    """

    async def wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        messages = list(kwargs.get("messages") or [])
        model = str(kwargs.get("model", "unknown"))
        is_stream = bool(kwargs.get("stream", False))

        normalized = normalize(OPENAI_CHAT_ASYNC, kwargs)
        input_result = guard.check(
            guard_input=normalized,
            event_type="input",
            model=model,
            llm_provider="openai",
        )

        if input_result and config.mode == "enforce":
            if input_result.blocked:
                raise TidewallBlockedError(
                    input_result.summary, input_result.detectors
                )
            if input_result.transformed and input_result.guard_output:
                kwargs["messages"] = input_result.guard_output["messages"]

        with gen_ai_span(provider="openai", model=model, guard_input=normalized) as span:
            response = await wrapped(*args, **kwargs)

            if not is_stream:
                try:
                    record_response_in_span(
                        span, content=response.choices[0].message.content
                    )
                except (IndexError, AttributeError, KeyError):
                    pass

        return response

    return wrapper
