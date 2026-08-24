"""The bound-call primitive. Task 3b of the P0 remediation plan."""

import pytest

from tidewall_otel._bound import bound_nodes, matches


def fake_create(self, *, messages, model, response_format=None, **kwargs):
    """Signature-faithful stand-in: keyword-only, with a `self` and a
    var-keyword, so the walk's handling of each is exercised."""


class _Omit:
    """Stands in for an SDK's unset sentinel."""


OMIT = _Omit()


# -- the pattern algebra --------------------------------------------------

@pytest.mark.parametrize("pattern,path,expected", [
    ("messages[*].content",  "messages[0].content",    True),
    ("messages[*].content",  "messages[7].content",    True),
    ("messages[*].content",  "messages[0].role",       False),
    ("messages[*].content",  "messages[0].content[0]", False),   # segment count
    ("messages[1:]",         "messages[1]",            True),
    ("messages[1:]",         "messages[0]",            False),   # the head
    ("messages[1:].role",    "messages[3].role",       True),
    ("messages[1:].role",    "messages[0].role",       False),
    ("messages[1:][*].role", "messages[3].role",       False),   # a broken form
    ("system",               "system",                 True),
])
def test_the_pattern_algebra(pattern, path, expected):
    assert matches(pattern, path) is expected


# -- the walk -------------------------------------------------------------

def test_positional_arguments_are_walked_too():
    """bind() exists precisely so positional args are not invisible; a walk
    over kwargs alone misses them entirely."""
    def positional(self, messages, model): ...
    nodes = bound_nodes(positional, (["msg"], "gpt-4o"), {})
    assert "messages" in nodes


def test_var_keyword_arguments_are_walked():
    """Anything arriving through **kwargs is unmapped and therefore lossy,
    not ignored. Tested against a callable that HAS one -- neither real SDK
    method does, which is why the fail-closed default matters on SDK upgrade
    rather than for unknown kwargs today."""
    nodes = bound_nodes(fake_create, (), {"messages": [], "model": "m",
                                          "surprise": "x"})
    assert "surprise" in nodes


def test_defaults_are_applied_before_the_walk():
    """A parameter left at its default is still provider-bound; omitting it
    hides a node from the equivalence."""
    nodes = bound_nodes(fake_create, (), {"messages": [], "model": "m"})
    assert "response_format" in nodes


def test_self_is_never_walked():
    """It is in ba.arguments, and an unaccounted node is lossy -- so walking
    `self` would make every call lossy."""
    nodes = bound_nodes(fake_create, (), {"messages": [], "model": "m"})
    assert "self" not in nodes


def test_bind_supplies_a_self_placeholder():
    """signature(...).bind(...) without a receiver raises TypeError on an
    unbound method resolved off the class, which is how the manifest resolves
    every surface."""
    nodes = bound_nodes(fake_create, (), {"messages": [], "model": "m"})
    assert nodes                                    # bound at all
    assert "messages" in nodes


def test_unsupplied_parameters_are_DROPPED():
    """The real SDK method binds 39 arguments for a two-key call, 33 of them
    an Omit sentinel. Keeping them makes every unclassified parameter refuse
    every call -- the fail-closed rule inverted into a denial of service."""
    nodes = bound_nodes(fake_create, (), {"messages": [], "model": "m",
                                          "response_format": OMIT},
                        unset=(_Omit,))
    assert "response_format" not in nodes
    assert set(nodes) >= {"messages", "model"}


def test_the_walk_produces_indexed_concrete_paths():
    """The exact keys every downstream predicate must match against."""
    nodes = bound_nodes(fake_create, (), {"model": "m", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": [{"type": "text", "text": "yo"}]},
    ]})
    assert "messages[0].content" in nodes
    assert "messages[1].content[0].type" in nodes
    assert nodes["messages[1].content"] == [{"type": "text", "text": "yo"}]


def test_declared_stops_are_not_descended_into():
    """A tool's JSON Schema is sent whole and inspected whole. Walking it
    produces paths no pattern can match, in BOTH bijection directions."""
    nodes = bound_nodes(fake_create, (), {"messages": [], "model": "m",
        "tools": [{"function": {"name": "w", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}}}}]},
        stop=("tools[*].function.parameters",))
    assert "tools[0].function.parameters" in nodes
    assert not any(k.startswith("tools[0].function.parameters.") for k in nodes)


def test_the_walk_descends_into_an_OBJECT_by_attribute():
    """A response object fed back into the next request is pydantic: it has
    no .get, so a walk written for mappings alone sees nothing inside it."""
    class Message:
        def __init__(self):
            self.role = "assistant"
            self.content = "the prior reply"

    nodes = bound_nodes(fake_create, (), {"model": "m",
                                          "messages": [Message()]})
    assert nodes["messages[0].content"] == "the prior reply"
    assert nodes["messages[0].role"] == "assistant"


# -- against the real SDKs ------------------------------------------------

from tests._fixtures import _sdk_installed

requires_openai = pytest.mark.skipif(
    not _sdk_installed("openai"), reason="openai not installed"
)


@requires_openai
def test_a_minimal_REAL_call_keeps_only_what_was_supplied():
    """The measurement the fail-closed rule depends on.

    A two-key call to the real method binds dozens of arguments, nearly all
    of them the SDK's not-supplied sentinel. If those are walked, every
    parameter not yet classified makes every call lossy, and enforce mode
    refuses all traffic.
    """
    import openai
    from openai.resources.chat.completions.completions import Completions

    call = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"}
    raw = bound_nodes(Completions.create, (), call)
    kept = bound_nodes(Completions.create, (), call,
                       unset=(openai._types.Omit, openai._types.NotGiven))

    assert len(raw) > 30, "the SDK binds far more than was supplied"
    assert len(kept) < 12, f"sentinels were not dropped: {sorted(kept)}"
    assert "self" not in kept
    assert kept["messages[0].content"] == "hi"
    assert set(kept) >= {"messages", "messages[0]", "messages[0].role", "model"}
