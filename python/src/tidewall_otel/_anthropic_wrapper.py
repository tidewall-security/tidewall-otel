"""Anthropic SDK ``wrapt`` wrappers for Tidewall guard integration.

Wraps :py:meth:`anthropic.resources.messages.messages.Messages.create`
(and the async variant) to inject inline Tidewall guard calls before
the underlying request is dispatched to the Anthropic API.

Differences from the OpenAI wrappers:

- Anthropic puts the system prompt in a separate ``system`` kwarg rather
  than as a ``role: "system"`` message. The normalizer reassembles these
  into a single OpenAI-shape message list for the guard.
- Anthropic content blocks are list-of-dict rather than plain strings.
"""

from __future__ import annotations

import logging
from typing import Any

from tidewall_otel._config import TidewallConfig
from tidewall_otel._exceptions import TidewallBlockedError
from tidewall_otel._guard import TidewallGuard
from tidewall_otel._normalizer import (
    extract_anthropic_response_text,
    normalize_anthropic_messages,
)
from tidewall_otel._span_helper import gen_ai_span, record_response_in_span

logger = logging.getLogger("tidewall.otel.anthropic")


def make_anthropic_sync_wrapper(
    guard: TidewallGuard, config: TidewallConfig
) -> Any:
    """Build a wrapt-compatible sync wrapper for ``Messages.create``."""

    def wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        model = str(kwargs.get("model", "unknown"))
        is_stream = bool(kwargs.get("stream", False))

        # --- INPUT GUARD ---
        # Normalize: pull the system kwarg + messages into OpenAI format.
        normalized = normalize_anthropic_messages(kwargs)
        input_result = guard.check(
            messages=normalized,
            event_type="input",
            model=model,
            llm_provider="anthropic",
        )

        if input_result and config.mode == "enforce":
            if input_result.blocked:
                raise TidewallBlockedError(
                    input_result.summary, input_result.detectors
                )
            if input_result.transformed and input_result.guard_output:
                _apply_transformed_input(
                    kwargs, input_result.guard_output["messages"]
                )
                logger.info(
                    "Input transformed by Tidewall (sensitive data redacted)"
                )

        # --- OTel SPAN + ORIGINAL CALL ---
        with gen_ai_span(
            provider="anthropic", model=model, messages=normalized
        ) as span:
            response = wrapped(*args, **kwargs)

            if not is_stream:
                try:
                    record_response_in_span(
                        span, content=extract_anthropic_response_text(response)
                    )
                except (IndexError, AttributeError, KeyError):
                    pass

        return response

    return wrapper


def make_anthropic_async_wrapper(
    guard: TidewallGuard, config: TidewallConfig
) -> Any:
    """Build a wrapt-compatible async wrapper for ``AsyncMessages.create``."""

    async def wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        model = str(kwargs.get("model", "unknown"))
        is_stream = bool(kwargs.get("stream", False))

        normalized = normalize_anthropic_messages(kwargs)
        input_result = guard.check(
            messages=normalized,
            event_type="input",
            model=model,
            llm_provider="anthropic",
        )

        if input_result and config.mode == "enforce":
            if input_result.blocked:
                raise TidewallBlockedError(
                    input_result.summary, input_result.detectors
                )
            if input_result.transformed and input_result.guard_output:
                _apply_transformed_input(
                    kwargs, input_result.guard_output["messages"]
                )

        with gen_ai_span(
            provider="anthropic", model=model, messages=normalized
        ) as span:
            response = await wrapped(*args, **kwargs)

            if not is_stream:
                try:
                    record_response_in_span(
                        span, content=extract_anthropic_response_text(response)
                    )
                except (IndexError, AttributeError, KeyError):
                    pass

        return response

    return wrapper


def _apply_transformed_input(
    kwargs: dict[str, Any], transformed_messages: list[dict[str, str]]
) -> None:
    """Apply Tidewall-transformed messages back into Anthropic's kwargs shape.

    The guard returns messages in OpenAI format (with a ``role: "system"``
    entry); Anthropic wants the system prompt as a separate kwarg. This
    helper splits them back out.
    """
    if not transformed_messages:
        return

    if transformed_messages[0].get("role") == "system":
        kwargs["system"] = transformed_messages[0]["content"]
        kwargs["messages"] = [
            {"role": m["role"], "content": m["content"]}
            for m in transformed_messages[1:]
        ]
    else:
        kwargs["messages"] = [
            {"role": m["role"], "content": m["content"]}
            for m in transformed_messages
        ]


def _mutate_anthropic_response(response: Any, new_content: str) -> None:
    """Replace the text content in an Anthropic ``Message`` response.

    Anthropic responses have a ``content`` list of typed blocks, e.g.
    ``[ContentBlock(type="text", text="...")]``. This helper updates the
    text on the first block in place when the underlying model is mutable.
    Frozen pydantic models are handled via ``model_copy``.
    """
    content_blocks = getattr(response, "content", [])
    if content_blocks and len(content_blocks) > 0:
        first_block = content_blocks[0]
        if hasattr(first_block, "text"):
            try:
                first_block.text = new_content
            except (AttributeError, TypeError):
                if hasattr(first_block, "model_copy"):
                    content_blocks[0] = first_block.model_copy(
                        update={"text": new_content}
                    )
