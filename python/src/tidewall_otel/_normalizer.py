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

      THIS FLATTENING DOES NOT MAKE SUCH A CALL INSPECTABLE. The manifest
      declares no ``content[*]`` paths, so a block list is LOSSY: ``enforce``
      refuses it before contacting the guard, and ``monitor``/``dry-run``
      proceed with a recorded skip. This function is reached in those modes
      and for the string case.

      Declaring the block paths was tried and reverted. Three provider paths
      (``content``, ``content[*].text``, ``content[*]``) would collapse onto
      one guard path, breaking the map's bijection -- an invariant that
      exists so a value cannot be inspected under another's name -- and the
      write-back cannot rebuild a block list from a redacted string, so a
      transform verdict would silently change the request's shape.

      The deeper reason is not mechanical: an image block carries
      instructions the guard cannot read. Flattening it away and reporting
      the call covered would claim an inspection that never happened.
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


# -- the guard-input contract (Task 12a) ----------------------------------

def _as_dict(value):
    """Objects -> plain dicts.

    SDK *input* types are TypedDicts, so tools supplied the ordinary way are
    already dicts and take the fast path. This branch exists for pydantic
    RESPONSE objects fed back into a request, which is the only way a real
    model instance reaches here.
    """
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python", exclude_unset=False, by_alias=False)
    return {name: item for name, item in vars(value).items()
            if not name.startswith("_")}


def _openai_tools(tools) -> list[dict]:
    """OpenAI already uses the guard's shape; normalise SDK models to dicts
    and keep only the mapped nodes."""
    out = []
    for tool in tools or ():
        entry = _as_dict(tool)
        function = _as_dict(entry.get("function") or {})
        out.append({"function": {
            "name": function.get("name"),
            "description": function.get("description"),
            "parameters": function.get("parameters"),
        }})
    return out


def _anthropic_tools(tools) -> list[dict]:
    """Anthropic is FLAT and differently named. This is the conversion the
    manifest's tools[*].input_schema -> guard_input.tools[*].function.parameters
    mapping asserts, and the server really does read that tree."""
    out = []
    for tool in tools or ():
        entry = _as_dict(tool)
        out.append({"function": {
            "name": entry.get("name"),
            "description": entry.get("description"),
            "parameters": entry.get("input_schema"),
        }})
    return out


def normalize(surface, kwargs: dict) -> dict:
    """The complete ``guard_input`` for one bound call.

    Emits ONLY what the server route reads -- ``messages`` and ``tools``.
    Emitting a field nothing inspects would put uninspected data on the wire
    and invite exactly the false certification the manifest exists to prevent,
    so ``tool_choice`` and ``response_format`` are deliberately absent.

    Message content is always a STRING: the route joins content across
    messages, so a typed list is a TypeError before any detector runs. Typed
    input is refused at dispatch, but the normalizer must still not emit a
    list, or a monitor-mode call would break the request it is only meant to
    observe.
    """
    if surface.provider == "anthropic":
        messages = normalize_anthropic_messages(kwargs)
        # The head message is ALWAYS emitted, with empty content when no
        # system prompt was supplied. Emitting it conditionally makes the
        # manifest's messages[1:] mapping false for every system-less call,
        # and makes any write-back assuming a head message delete a real turn.
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": ""}, *messages]
        tools = _anthropic_tools(kwargs.get("tools"))
    else:
        messages = normalize_openai_messages(kwargs.get("messages") or [])
        tools = _openai_tools(kwargs.get("tools"))

    return {"messages": messages, "tools": tools}
