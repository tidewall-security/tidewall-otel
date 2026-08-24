"""The wire contract. Task 12a of the P0 remediation plan."""

import inspect

import pytest

from tests._fixtures import _sdk_installed
from tidewall_otel._manifest import (
    ANTHROPIC_MESSAGES_SYNC,
    OPENAI_CHAT_SYNC,
    SURFACES,
)
from tidewall_otel._normalizer import normalize

requires_sdks = pytest.mark.skipif(
    not (_sdk_installed("openai") and _sdk_installed("anthropic")),
    reason="provider SDKs not installed",
)

OPENAI_CALL = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Look up the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
}

ANTHROPIC_CALL = {
    "model": "claude-x",
    "max_tokens": 8,
    "system": "be helpful",
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [{"name": "get_weather", "description": "Look up the weather",
               "input_schema": {"type": "object",
                                "properties": {"city": {"type": "string"}}}}],
}


def call_for(surface):
    return OPENAI_CALL if surface.provider == "openai" else ANTHROPIC_CALL


# -- what goes on the wire ------------------------------------------------

@pytest.mark.parametrize("surface", SURFACES, ids=lambda s: s.attribute)
def test_the_normalizer_emits_NOTHING_the_route_cannot_read(surface):
    """The server reads guard_input.messages and guard_input.tools and nothing
    else. Emitting a field nothing inspects puts uninspected data on the wire
    and invites the false certification this contract exists to prevent."""
    guard_input = normalize(surface, call_for(surface))
    assert set(guard_input) <= {"messages", "tools"}, guard_input


@pytest.mark.parametrize("surface", SURFACES, ids=lambda s: s.attribute)
def test_message_content_on_the_wire_is_always_a_STRING(surface):
    """The route does `" ".join(m.get("content", ""))`, so a typed list is a
    TypeError before any detector runs."""
    guard_input = normalize(surface, call_for(surface))
    for message in guard_input["messages"]:
        assert isinstance(message["content"], str), message


def test_the_sender_transmits_TOOLS_not_only_messages():
    """The route reads guard_input.tools, and the client never wrote it."""
    guard_input = normalize(OPENAI_CHAT_SYNC, OPENAI_CALL)
    assert "tools" in guard_input and guard_input["tools"]


def test_the_openai_tool_tree_is_sent_whole():
    tools = normalize(OPENAI_CHAT_SYNC, OPENAI_CALL)["tools"]
    assert tools[0]["function"]["name"] == "get_weather"
    assert tools[0]["function"]["parameters"]["properties"]["city"]["type"] == "string"


def test_the_anthropic_tool_schema_is_CONVERTED_not_merely_present():
    """Anthropic is flat and differently named, and the server reads this
    tree, so the values are asserted rather than the key's presence."""
    tools = normalize(ANTHROPIC_MESSAGES_SYNC, ANTHROPIC_CALL)["tools"]
    assert tools == [{"function": {
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}}},
    }}]


def test_the_anthropic_head_message_is_ALWAYS_emitted():
    """Emitting it only when `system` is truthy makes the manifest's
    messages[1:] mapping false for every system-less call, and makes any
    write-back assuming a head message delete a real user turn."""
    without_system = dict(ANTHROPIC_CALL)
    del without_system["system"]

    messages = normalize(ANTHROPIC_MESSAGES_SYNC, without_system)["messages"]
    assert messages[0] == {"role": "system", "content": ""}
    assert messages[1]["content"] == "hi"


def test_the_anthropic_system_prompt_becomes_the_head_message():
    messages = normalize(ANTHROPIC_MESSAGES_SYNC, ANTHROPIC_CALL)["messages"]
    assert messages[0] == {"role": "system", "content": "be helpful"}


def test_typed_content_is_FLATTENED_to_its_text():
    """Typed input is refused at dispatch; the normalizer must still not
    produce a list, or a monitor-mode call would break the request."""
    call = {**OPENAI_CALL, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "a"},
                                     {"type": "text", "text": "b"}]}]}
    guard_input = normalize(OPENAI_CHAT_SYNC, call)
    assert guard_input["messages"][0]["content"] == "a b"


# -- every caller migrated in the same commit -----------------------------

def test_check_takes_GUARD_INPUT_and_still_returns_a_GuardResult():
    """12a is a PURE INPUT-CONTRACT migration: the four wrapper bodies keep
    their control flow and dereference GuardResult attributes, so changing the
    RETURN type here would be an AttributeError in all four."""
    from tidewall_otel._guard import TidewallGuard

    parameters = inspect.signature(TidewallGuard.check).parameters
    assert "guard_input" in parameters
    assert "messages" not in parameters

    returns = inspect.signature(TidewallGuard.check).return_annotation
    assert "GuardResult" in str(returns), returns


def test_the_span_helper_was_migrated_too():
    """It receives the same value; a caller still passing `messages=` would
    silently accept the wrong shape rather than raising."""
    from tidewall_otel._span_helper import gen_ai_span

    parameters = inspect.signature(gen_ai_span).parameters
    assert "guard_input" in parameters
    assert "messages" not in parameters


def test_payload_for_wraps_guard_input_in_the_envelope():
    """Task 3 owns post_guard and its `payload`; 12a produces the envelope
    builder that turns a guard input into that payload."""
    from tidewall_otel._config import TidewallConfig
    from tidewall_otel._guard import TidewallGuard

    guard = TidewallGuard(TidewallConfig())
    payload = guard._payload_for({"messages": [], "tools": []})
    assert payload["guard_input"] == {"messages": [], "tools": []}
    assert "event_type" in payload and "app_id" in payload
