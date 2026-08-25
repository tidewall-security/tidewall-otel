"""The supported-surface manifest. Task 4 of the P0 remediation plan."""

import inspect

import httpx
import pytest

from tests._fixtures import _sdk_installed
from tidewall_otel._bound import bound_nodes, matches
from tidewall_otel._manifest import (
    ANTHROPIC_MESSAGES_ASYNC,
    ANTHROPIC_MESSAGES_SYNC,
    OPENAI_CHAT_ASYNC,
    OPENAI_CHAT_SYNC,
    OUT_OF_SCOPE,
    SURFACES,
    _is_standard_transport,
    client_escapes,
    disposition_for,
    lossy_paths,
    resolve,
)

requires_sdks = pytest.mark.skipif(
    not (_sdk_installed("openai") and _sdk_installed("anthropic")),
    reason="provider SDKs not installed",
)


def bound_nodes_for(surface, args, kwargs, wrapped=None):
    """The ONLY way this suite walks a call, so no test can exercise the walk
    without the stops and sentinels that keep the fail-closed rule from
    refusing everything."""
    return bound_nodes(
        wrapped or resolve(surface), args, kwargs,
        stop=surface.opaque_subtrees + surface.non_prompt_bearing,
        unset=surface.unset_sentinels,
    )


# -- the contract ---------------------------------------------------------

def test_every_currently_patched_method_is_in_the_manifest():
    """Anything patched and unlisted is a boundary with no contract."""
    from tidewall_otel._instrumentor import _ANTHROPIC_MODULE, _OPENAI_MODULE

    # DERIVE what activation actually patches and compare the whole set.
    # Naming four members cannot detect a FIFTH: adding
    # `(_OPENAI_MODULE, "Completions.stream", ...)` to the install specs left
    # this test green while a boundary with no contract was patched -- which
    # is exactly what its docstring says it prevents.
    import tidewall_otel._instrumentor as instrumentor_module
    from tidewall_otel._config import TidewallConfig
    from tidewall_otel._manager import PatchManager

    installed: list[tuple[str, str]] = []
    real_install_all = PatchManager.install_all

    def record(self, specs, *args, **kwargs):
        installed.extend((module, attribute) for module, attribute, _ in specs)
        return real_install_all(self, specs, *args, **kwargs)

    instrumentor = instrumentor_module.TidewallInstrumentor()
    PatchManager.install_all = record
    try:
        instrumentor._instrument(config=TidewallConfig(
            base_url="https://guard.example", token="t"))
    finally:
        PatchManager.install_all = real_install_all
        instrumentor._uninstrument()

    listed = {(surface.module, surface.attribute) for surface in SURFACES}
    assert installed, "activation patched nothing, so this proves nothing"
    assert set(installed) <= listed, (
        f"patched but unlisted: {sorted(set(installed) - listed)}")


def test_every_entry_declares_a_complete_contract():
    for surface in SURFACES:
        assert surface.provider_fields, surface
        assert surface.path_map, surface
        assert surface.lossless_for, surface
        assert surface.lossless_shapes, surface
        assert surface.unset_sentinel_refs, surface
        assert surface.version_range, surface
        assert surface.span_input is False and surface.span_output is False, (
            f"{surface.attribute}: span content must default OFF"
        )


def test_the_path_map_is_bijective():
    """No two provider paths may map to one guard path, or a value could be
    inspected under another's name."""
    for surface in SURFACES:
        targets = list(surface.path_map.values())
        assert len(targets) == len(set(targets)), surface.attribute


def test_no_entry_maps_a_field_the_route_does_not_read():
    """The server reads guard_input.messages and guard_input.tools and
    NOTHING else. A mapping to any other key certifies an inspection that
    does not happen, and both bijection directions compare client-side paths,
    so nothing else can detect it."""
    READ_BY_THE_ROUTE = ("guard_input.messages", "guard_input.tools")
    for surface in SURFACES:
        for target in surface.path_map.values():
            assert target.startswith(READ_BY_THE_ROUTE), (
                f"{surface.attribute}: maps {target}, which no route reads"
            )


def test_unlisted_surfaces_are_named_as_out_of_scope():
    assert "openai.responses" in OUT_OF_SCOPE
    assert "anthropic.messages.stream" in OUT_OF_SCOPE


def test_surface_entries_are_frozen_hashable_and_NOT_slotted():
    """cached_property writes into __dict__, bypassing the frozen __setattr__.
    Adding slots=True raises at first access, and slots is an obvious-looking
    optimisation for a frozen dataclass."""
    import dataclasses

    for surface in SURFACES:
        assert dataclasses.is_dataclass(surface)
        assert hash(surface) is not None
        assert hasattr(surface, "__dict__"), (
            f"{surface.attribute}: must not use slots=True"
        )


def test_importing_the_manifest_imports_NO_sdk():
    """Both SDKs are optional extras; the manifest must import without them."""
    from tests._fixtures import run_python

    out = run_python("-c", "import sys, tidewall_otel._manifest; "
                           "print('openai' in sys.modules, 'anthropic' in sys.modules)")
    assert out.strip() == "False False"


@requires_sdks
def test_unset_sentinels_resolve_to_TYPES_when_the_sdk_is_present():
    for surface in SURFACES:
        for sentinel in surface.unset_sentinels:
            assert isinstance(sentinel, type), (surface.attribute, sentinel)
        assert isinstance(object(), surface.unset_sentinels) is False


# -- lossiness, by predicate rather than by label -------------------------

@requires_sdks
def test_a_minimal_real_call_has_NO_lossy_paths():
    """The regression test for the fail-closed inversion. If this fails,
    enforce mode blocks all traffic."""
    nodes = bound_nodes_for(OPENAI_CHAT_SYNC, (), {
        "messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"})
    assert lossy_paths(OPENAI_CHAT_SYNC, nodes) == ()


@requires_sdks
def test_typed_content_is_lossy_BY_PREDICATE_not_by_label():
    """The server joins message content as strings, so a typed list is a
    TypeError before any detector runs. Asserting two English labels are
    absent from a tuple would pass with the mechanism ignored entirely."""
    nodes = bound_nodes_for(OPENAI_CHAT_SYNC, (), {"model": "gpt-4o", "messages": [
        {"role": "user", "content": "plain"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]})
    lossy = lossy_paths(OPENAI_CHAT_SYNC, nodes)
    assert "messages[1].content" in lossy
    assert "messages[0].content" not in lossy


@requires_sdks
def test_an_anthropic_TYPED_SYSTEM_BLOCK_is_lossy():
    target = resolve(ANTHROPIC_MESSAGES_SYNC)
    typed = bound_nodes_for(ANTHROPIC_MESSAGES_SYNC, (), {
        "system": [{"type": "text", "text": "be helpful"}],
        "messages": [], "model": "claude-x", "max_tokens": 8}, wrapped=target)
    assert "system" in lossy_paths(ANTHROPIC_MESSAGES_SYNC, typed)

    plain = bound_nodes_for(ANTHROPIC_MESSAGES_SYNC, (), {
        "system": "be helpful", "messages": [], "model": "claude-x",
        "max_tokens": 8}, wrapped=target)
    assert "system" not in lossy_paths(ANTHROPIC_MESSAGES_SYNC, plain)


@requires_sdks
@pytest.mark.parametrize("surface", [OPENAI_CHAT_SYNC, ANTHROPIC_MESSAGES_SYNC],
                         ids=lambda s: s.attribute)
def test_a_nonempty_extra_body_is_LOSSY(surface):
    """O-11. Both SDKs merge extra_body OVER the generated body, so it can
    replace the very messages the guard just approved."""
    base = ({"messages": [], "model": "gpt-4o"} if surface is OPENAI_CHAT_SYNC
            else {"messages": [], "model": "c", "max_tokens": 8})

    override = bound_nodes_for(surface, (), {
        **base, "extra_body": {"messages": [{"role": "user", "content": "EVIL"}]}})
    assert "extra_body" in lossy_paths(surface, override)

    absent = bound_nodes_for(surface, (), base)
    assert "extra_body" not in lossy_paths(surface, absent)


@requires_sdks
def test_an_allowed_tools_tool_choice_is_lossy_and_a_named_one_is_not():
    """Three of tool_choice's four variants are pure controls; the fourth
    supplies INLINE tool objects the guard never sees."""
    inline = bound_nodes_for(OPENAI_CHAT_SYNC, (), {
        "messages": [], "model": "gpt-4o",
        "tool_choice": {"type": "allowed_tools", "allowed_tools": {
            "mode": "auto", "tools": [{"function": {"name": "exfiltrate"}}]}}})
    assert "tool_choice" in lossy_paths(OPENAI_CHAT_SYNC, inline)

    for control in ("auto", "required",
                    {"type": "function", "function": {"name": "get_weather"}}):
        nodes = bound_nodes_for(OPENAI_CHAT_SYNC, (), {
            "messages": [], "model": "gpt-4o", "tool_choice": control})
        assert "tool_choice" not in lossy_paths(OPENAI_CHAT_SYNC, nodes), control


@requires_sdks
def test_an_anthropic_output_config_SCHEMA_is_lossy_and_effort_only_is_not():
    """OutputConfigParam.format.schema is Dict[str, object] -- descriptions,
    enums and examples no detector sees. Classified from its TYPE, not its
    name."""
    base = {"messages": [], "model": "c", "max_tokens": 8}
    with_schema = bound_nodes_for(ANTHROPIC_MESSAGES_SYNC, (), {
        **base, "output_config": {"format": {"type": "json_schema",
                                             "schema": {"description": "secret"}}}})
    assert "output_config" in lossy_paths(ANTHROPIC_MESSAGES_SYNC, with_schema)

    effort_only = bound_nodes_for(ANTHROPIC_MESSAGES_SYNC, (), {
        **base, "output_config": {"effort": "high"}})
    assert "output_config" not in lossy_paths(ANTHROPIC_MESSAGES_SYNC, effort_only)


@requires_sdks
def test_a_json_schema_response_format_is_lossy_and_a_plain_one_is_not():
    schema = bound_nodes_for(OPENAI_CHAT_SYNC, (), {
        "messages": [], "model": "gpt-4o",
        "response_format": {"type": "json_schema",
                            "json_schema": {"description": "secret"}}})
    assert "response_format" in lossy_paths(OPENAI_CHAT_SYNC, schema)

    plain = bound_nodes_for(OPENAI_CHAT_SYNC, (), {
        "messages": [], "model": "gpt-4o",
        "response_format": {"type": "json_object"}})
    assert "response_format" not in lossy_paths(OPENAI_CHAT_SYNC, plain)


@requires_sdks
def test_the_legacy_functions_API_is_lossy_not_silently_uninspected():
    """functions[*] carries name, description and parameters and is never
    sent. Classifying it a control would certify an inspection that does not
    happen."""
    nodes = bound_nodes_for(OPENAI_CHAT_SYNC, (), {
        "messages": [], "model": "gpt-4o",
        "functions": [{"name": "run_sql", "description": "runs SQL",
                       "parameters": {"type": "object"}}]})
    assert "functions" in lossy_paths(OPENAI_CHAT_SYNC, nodes)


def test_an_unknown_kwarg_is_lossy_without_any_declaration():
    """The fail-closed default, through the same function."""
    def future_sdk(self, *, messages, model, **kwargs): ...

    nodes = bound_nodes_for(OPENAI_CHAT_SYNC, (), {
        "messages": [], "model": "gpt-4o", "some_new_2027_field": "x"},
        wrapped=future_sdk)
    assert "some_new_2027_field" in lossy_paths(OPENAI_CHAT_SYNC, nodes)


# -- signature classification --------------------------------------------

@requires_sdks
def test_every_signature_parameter_is_CLASSIFIED():
    """Three categories: mapped, non_prompt_bearing, or known_lossy.

    EXPECTED TO BREAK ON SDK UPGRADE. That is the point: a new parameter must
    be classified deliberately, not silently uninspected.
    """
    root = lambda pattern: pattern.split("[")[0].split(".")[0]
    for surface in SURFACES:
        params = set(inspect.signature(resolve(surface)).parameters) - {"self", "kwargs", "args"}
        mapped = {root(p) for p in surface.path_map}
        declared = ({root(p) for p in surface.non_prompt_bearing}
                    | {root(p) for p in surface.known_lossy})
        unaccounted = params - mapped - declared
        assert not unaccounted, (
            f"{surface.attribute}: {sorted(unaccounted)} are in no category. "
            f"An SDK upgrade added parameters -- classify each as mapped, "
            f"non_prompt_bearing, or known_lossy. Do NOT bulk-add to "
            f"non_prompt_bearing: check what each one carries first."
        )
        contract = mapped | {root(p) for p in surface.known_lossy}
        broken = contract - params
        assert not broken, (
            f"{surface.attribute}: {sorted(broken)} are mapped or declared "
            f"lossy but absent from this SDK -- the contract is dead code"
        )


@requires_sdks
def test_no_non_prompt_bearing_entry_hides_a_nested_schema():
    """The GENERAL form, so the next one is caught by the suite rather than a
    reviewer. eval_str is load-bearing: both SDKs use postponed annotations,
    so without it every annotation is a string and this flags nothing."""
    from tests._fixtures import _admits_free_form

    for surface in SURFACES:
        signature = inspect.signature(resolve(surface), eval_str=True)

        # SANITY: known cases must flag, or this proves nothing.
        assert _admits_free_form(signature.parameters["extra_body"].annotation)

        for name in surface.non_prompt_bearing:
            if name not in signature.parameters:
                continue
            if _admits_free_form(signature.parameters[name].annotation):
                assert _shape_for_exists(surface, name), (
                    f"{surface.attribute}.{name} transitively admits "
                    f"Dict[str, object] but is declared a pure control with "
                    f"no predicate"
                )


def _shape_for_exists(surface, name):
    return any(matches(pattern, name) for pattern in surface.lossless_shapes)


# -- client integrity -----------------------------------------------------

@requires_sdks
@pytest.mark.parametrize("label,construct", [
    ("openai default",        lambda: __import__("openai").OpenAI(api_key="t")),
    ("openai async",          lambda: __import__("openai").AsyncOpenAI(api_key="t")),
    ("anthropic default",     lambda: __import__("anthropic").Anthropic(api_key="t")),
    ("anthropic async",       lambda: __import__("anthropic").AsyncAnthropic(api_key="t")),
    ("openai proxy",          lambda: __import__("openai").OpenAI(
        api_key="t", http_client=httpx.Client(proxy="http://localhost:8888"))),
    ("openai async proxy",    lambda: __import__("openai").AsyncOpenAI(
        api_key="t", http_client=httpx.AsyncClient(proxy="http://localhost:8888"))),
    ("anthropic proxy",       lambda: __import__("anthropic").Anthropic(
        api_key="t", http_client=httpx.Client(proxy="http://localhost:8888"))),
    ("custom timeout",        lambda: __import__("openai").OpenAI(api_key="t", timeout=30.0)),
    ("base_url override",     lambda: __import__("openai").OpenAI(
        api_key="t", base_url="https://gateway.internal/v1")),
    ("max_retries",           lambda: __import__("openai").OpenAI(api_key="t", max_retries=5)),
    ("default_headers",       lambda: __import__("openai").OpenAI(
        api_key="t", default_headers={"x-team": "search"})),
    ("verify=False",          lambda: __import__("openai").OpenAI(
        api_key="t", http_client=httpx.Client(verify=False))),
    ("custom limits",         lambda: __import__("openai").OpenAI(
        api_key="t", http_client=httpx.Client(limits=httpx.Limits(max_connections=5)))),
    ("trust_env=False",       lambda: __import__("openai").OpenAI(
        api_key="t", http_client=httpx.Client(trust_env=False))),
], ids=lambda x: x if isinstance(x, str) else "")
def test_an_ordinary_client_has_NO_escapes(label, construct):
    """The false-positive guard matters more than the positive one: a detector
    that fires on ordinary usage makes is_active() false for every proxy user
    and trains the operator to ignore it."""
    assert client_escapes(construct()) == (), label


class _Rewriter(httpx.BaseTransport):
    def handle_request(self, request):
        return httpx.Response(200, json={})


@requires_sdks
@pytest.mark.parametrize("construct,expected", [
    (lambda: __import__("anthropic").Anthropic(
        api_key="t", middleware=[lambda r, n: n(r)]), "middleware"),
    (lambda: __import__("openai").OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200)))), "transport"),
    (lambda: __import__("openai").OpenAI(api_key="t", http_client=httpx.Client(
        mounts={"all://": httpx.MockTransport(lambda r: httpx.Response(200))})), "mounts"),
    (lambda: __import__("openai").OpenAI(api_key="t", http_client=httpx.Client(
        event_hooks={"request": [lambda r: None]})), "event_hooks"),
])
def test_each_construction_time_escape_is_DETECTED(construct, expected):
    """Out of the threat model, but never unnoticed: each was reproduced
    rewriting the wire body after inspection."""
    assert expected in client_escapes(construct())


def test_a_transport_WEARING_THE_STANDARD_NAME_is_not_trusted():
    """Comparing class NAMES certifies an impostor. Identity, not names."""
    Impostor = type("HTTPTransport", (_Rewriter,), {})
    assert type(Impostor()).__name__ == "HTTPTransport"
    assert not _is_standard_transport(Impostor())


def test_a_SUBCLASS_of_the_real_transport_is_not_trusted_either():
    """Why identity and not isinstance: a subclass overriding handle_request
    passes isinstance and can rewrite the body freely."""
    class Sneaky(httpx.HTTPTransport):
        def handle_request(self, request):
            return httpx.Response(200, json={})

    assert isinstance(Sneaky(), httpx.HTTPTransport)
    assert not _is_standard_transport(Sneaky())


def test_genuine_transports_are_still_trusted_under_identity():
    """Tightening the check must not start flagging the real ones."""
    assert _is_standard_transport(httpx.Client()._transport)
    assert _is_standard_transport(httpx.AsyncClient()._transport)
    assert _is_standard_transport(None)
    for mounted in httpx.Client(proxy="http://localhost:8888")._mounts.values():
        assert _is_standard_transport(mounted)


# -- disposition -----------------------------------------------------------

def test_an_out_of_range_sdk_makes_the_surface_UNVERIFIED():
    assert disposition_for(OPENAI_CHAT_SYNC, installed_version="1.50.0") == "covered"
    assert disposition_for(OPENAI_CHAT_SYNC, installed_version="2.0.0") == "unverified"
    assert disposition_for(OPENAI_CHAT_SYNC, installed_version="1.0.0") == "unverified"


@requires_sdks
def test_the_extra_body_OVERRIDE_is_REAL_and_is_classified_lossy():
    """Two assertions, because the refusal is only meaningful if the threat is.

    First: capture the body the SDK actually builds, with no Tidewall in the
    picture, and prove extra_body wins. If a future SDK stops merging, this
    fails and the refusal can be reconsidered deliberately rather than left in
    place for a reason that stopped being true.

    Second: the manifest classifies it lossy. Refusal at dispatch is asserted
    where dispatch exists.
    """
    import json

    import openai

    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0,
            "model": "gpt-4o", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}]})

    client = openai.OpenAI(
        api_key="t", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "SAFE"}],
        extra_body={"messages": [{"role": "user", "content": "EVIL"}]},
    )
    assert captured["body"]["messages"][0]["content"] == "EVIL", (
        "extra_body no longer overrides the wire body -- re-examine O-11"
    )

    nodes = bound_nodes_for(OPENAI_CHAT_SYNC, (), {
        "model": "gpt-4o", "messages": [{"role": "user", "content": "SAFE"}],
        "extra_body": {"messages": [{"role": "user", "content": "EVIL"}]}})
    assert "extra_body" in lossy_paths(OPENAI_CHAT_SYNC, nodes)


def test_known_lossy_is_an_EXPLICIT_declaration_not_a_fallback():
    """Mutation-testing showed `known_lossy` adds no DETECTION power: its
    entries are in neither path_map nor non_prompt_bearing, so the unknown-key
    fallback catches them anyway, and deleting the known_lossy branch left
    every test passing.

    What it does provide is an explicit, reviewable statement that a
    prompt-bearing field was considered and refused, rather than silently
    falling through. That is worth having and worth asserting -- but the
    earlier claim that it is "stronger than the fallback" was simply false.
    """
    assert "functions" in OPENAI_CHAT_SYNC.known_lossy
    assert "prediction" in OPENAI_CHAT_SYNC.known_lossy
    for name in OPENAI_CHAT_SYNC.known_lossy:
        assert name not in OPENAI_CHAT_SYNC.non_prompt_bearing, (
            f"{name} is declared both a control and lossy"
        )


def test_the_anthropic_write_back_splits_the_system_prompt_BACK_OUT():
    """The transform must restore Anthropic's separate `system` kwarg."""
    kwargs = {"messages": [{"role": "user", "content": "hi"}],
              "system": "be careful", "model": "claude-x"}
    guard_messages = [{"role": "system", "content": "be careful, cleaned"},
                      {"role": "user", "content": "hi"}]

    out = ANTHROPIC_MESSAGES_SYNC.transform_into(kwargs, guard_messages)
    assert out["system"] == "be careful, cleaned"
    assert out["messages"] == [{"role": "user", "content": "hi"}]
    assert all(m["role"] != "system" for m in out["messages"])
    assert out["model"] == "claude-x", "untouched kwargs must survive"


def test_the_anthropic_write_back_KEEPS_every_message_when_there_is_no_system():
    """Taking the head unconditionally deletes the caller's FIRST MESSAGE on
    every ordinary call -- the normalizer only prepends a system message when
    `system` was supplied."""
    kwargs = {"messages": [{"role": "user", "content": "first"}], "model": "c"}
    guard_messages = [{"role": "user", "content": "first"},
                      {"role": "assistant", "content": "second"}]

    out = ANTHROPIC_MESSAGES_SYNC.transform_into(kwargs, guard_messages)
    assert "system" not in out

    # DERIVED from what was supplied. Asserting `len(...) == 2` and then
    # member [0] checks cardinality and one element -- a write-back that
    # corrupted every message after the first passed exactly that pair of
    # assertions under a name promising EVERY message.
    assert out["messages"] == guard_messages, (
        f"the write-back did not preserve every message: {out['messages']}")


def test_the_anthropic_write_back_tolerates_an_empty_list():
    out = ANTHROPIC_MESSAGES_SYNC.transform_into({"model": "c"}, [])
    assert out == {"model": "c"}


def test_the_openai_write_back_replaces_only_messages():
    kwargs = {"messages": [{"role": "user", "content": "old"}],
              "model": "gpt-4o", "temperature": 0.2, "seed": 7}
    guard_messages = [{"role": "user", "content": "clean"}]
    out = OPENAI_CHAT_SYNC.transform_into(kwargs, guard_messages)

    # WHOLE OBJECT. Checking `messages[0]` plus three named kwargs let a
    # write-back that APPENDED a fabricated assistant message pass -- the
    # first message was right and every named kwarg survived, and the test
    # never looked at what else was in the list.
    assert out == {**kwargs, "messages": guard_messages}
