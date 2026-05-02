"""Message format normalization between AI SDKs and the Tidewall guard.

The Tidewall guard expects messages in OpenAI Chat Completions shape —
a list of ``{"role": "...", "content": "..."}`` dicts. SDK-specific
formats (notably Anthropic's separate ``system`` kwarg and content
blocks) are converted to and from that canonical form here.

Keeping the conversion logic in one module makes it easier to add
support for additional providers (Cohere, Mistral, etc.) without
sprinkling format-specific code through the wrappers.
"""

from __future__ import annotations

from typing import Any


def normalize_openai_messages(
    messages: Any,
) -> list[dict[str, str]]:
    """Normalize OpenAI message list, handling pydantic models and multi-part content.

    OpenAI messages are already in the canonical shape, but this helper
    coerces edge cases:

    - Multi-part content (vision: ``[{type: "text", text: "..."}, ...]``)
      is reduced to a space-joined plain string of the text parts.
    - Pydantic model instances are unwrapped via attribute access.
    - ``role`` defaults to ``"user"`` if missing.
    """
    normalized = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part content (e.g. vision) — extract just the text parts.
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)
            normalized.append({"role": str(role), "content": str(content or "")})
        else:
            # Pydantic model or similar — fall back to attribute access.
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                text_parts = []
                for p in content:
                    if isinstance(p, dict):
                        text_parts.append(p.get("text", ""))
                    elif hasattr(p, "text"):
                        text_parts.append(str(p.text))
                content = " ".join(text_parts)
            normalized.append({"role": str(role), "content": str(content or "")})
    return normalized


def normalize_anthropic_messages(
    kwargs: dict[str, Any],
) -> list[dict[str, str]]:
    """Convert Anthropic ``create()`` kwargs into the canonical message list.

    Handles two Anthropic-specific shapes:

    - ``system`` is a separate kwarg (string or list of text blocks);
      we prepend it as a ``role: "system"`` message.
    - ``messages[].content`` may be a list of typed content blocks
      (``{type: "text", text: "..."}``); we flatten the text parts.
    """
    normalized: list[dict[str, str]] = []

    system = kwargs.get("system")
    if system:
        if isinstance(system, str):
            normalized.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, dict):
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    text_parts.append(str(block.text))
            normalized.append({"role": "system", "content": " ".join(text_parts)})

    messages = kwargs.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    text_parts.append(str(block.text))
            content = " ".join(text_parts)

        normalized.append({"role": str(role), "content": str(content or "")})

    return normalized


def extract_anthropic_response_text(response: Any) -> str:
    """Extract the text portion of an Anthropic ``Message`` response.

    Anthropic responses carry a ``content`` list of typed blocks
    (``[ContentBlock(type="text", text="...")]``). This helper joins
    the text parts together and ignores other block types (tool calls,
    images, etc.).
    """
    content = getattr(response, "content", [])
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(str(block.text))
        return " ".join(text_parts)
    return str(content)
