"""The supported-surface manifest: checked-in data, one entry per surface.

Each entry is a contract about a single patched SDK method -- what the agent
sends, what it deliberately does not, and what it cannot faithfully send at
all. Coverage is decided from this data rather than from what happens to be
patched, so a boundary with no entry is a boundary with no claim.

WHAT "LOSSY" MEANS HERE
-----------------------

Lossiness is an executable predicate over a bound value, not an English label.
``lossless_for`` is prose for humans; ``lossless_shapes`` is what
:func:`lossy_paths` evaluates and what dispatch acts on. A decision recorded
only in prose is contradicted by the data beside it sooner or later.

COVERAGE IS DEFINED OVER THE WIRE BODY
--------------------------------------

A parameter that can add to or override the request the provider actually
receives is *wire-rewriting*, and may never be classified non-prompt-bearing.
``extra_body`` is merged over the generated JSON body by both SDKs, so a call
can have one payload inspected and a different one sent. Establishing that a
particular payload is harmless would mean reimplementing the provider's merge
and its full request schema, so a non-empty one is refused outright.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from ._bound import matches

OUT_OF_SCOPE = (
    "openai.responses",             # a different API surface, not wrapped
    "anthropic.messages.stream",    # streaming is out of the commitment
)


# -- shape predicates ------------------------------------------------------

def _is_str(value: Any) -> bool:
    """Lossless content is a string.

    A list of typed parts fails, and so does an SDK model instance, which is
    also not a str -- one predicate covers both the dict-parts and the
    pydantic-object cases. The server joins message content as strings, so a
    typed list is a TypeError before any detector runs.
    """
    return isinstance(value, str)


def _is_empty_extra_body(value: Any) -> bool:
    """A NON-EMPTY ``extra_body`` is lossy. Unconditionally, both providers.

    Reproduced against openai 1.109.1 and anthropic 0.125.0: both merge
    ``extra_body`` over the generated JSON body, so ``messages=[SAFE]`` with
    ``extra_body={"messages": [EVIL]}`` puts EVIL on the wire while the guard
    inspected SAFE. There is deliberately no attempt to prove a particular
    payload harmless.
    """
    return not value


def _extra_query_cannot_carry_a_prompt(value: Any) -> bool:
    """``extra_query`` is flagged by the free-form scan and DELIBERATELY
    permitted, with the reason recorded here rather than by omitting it.

    Unlike ``extra_body`` it merges into the URL query string, not the JSON
    body, and both providers read the prompt exclusively from the body. If
    either API ever accepts prompt content by query parameter, this predicate
    is where that changes -- and the scan will keep pointing at it.
    """
    return True


def _is_toolless_tool_choice(value: Any) -> bool:
    """OpenAI ``tool_choice`` is a control for three of its four variants.

    ``allowed_tools`` is the exception: its ``allowed_tools.tools`` is
    ``Iterable[Dict[str, object]]`` -- inline tool objects carrying names and
    descriptions no detector sees. The named variants only reference tools
    already inspected under ``tools``.
    """
    if value is None or isinstance(value, str):
        return True
    kind = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return kind != "allowed_tools"


def _is_schemaless_output_config(value: Any) -> bool:
    """anthropic ``output_config`` is NOT a pure control.

    ``OutputConfigParam.format`` is a ``JSONOutputFormatParam`` whose
    ``schema`` is ``Dict[str, object]``: descriptions, enum strings, examples.
    The exact twin of OpenAI's ``response_format={"type": "json_schema"}``.
    """
    if value is None:
        return True
    fmt = value.get("format") if isinstance(value, dict) else getattr(value, "format", None)
    return fmt is None


def _is_schemaless_response_format(value: Any) -> bool:
    """``text`` and ``json_object`` carry no author text; ``json_schema``
    carries ``description`` strings no detector sees."""
    if value is None:
        return True
    kind = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return kind in ("text", "json_object")


# -- transform write-back --------------------------------------------------

def _openai_write_back(kwargs: dict, messages: list) -> dict:
    return {**kwargs, "messages": messages}


def _anthropic_write_back(kwargs: dict, messages: list) -> dict:
    """Split the guard's OpenAI-shaped list back into Anthropic's separate
    ``system`` kwarg plus ``messages``.

    Taking the head unconditionally would delete the caller's first message
    whenever no system prompt was present.
    """
    if not messages:
        return dict(kwargs)

    out = dict(kwargs)
    if messages[0].get("role") == "system":
        out["system"] = messages[0]["content"]
        rest = messages[1:]
    else:
        rest = messages
    out["messages"] = [{"role": m["role"], "content": m["content"]} for m in rest]
    return out


# -- client integrity ------------------------------------------------------

def _standard_transports() -> tuple[type, ...]:
    """The two httpx transports we can vouch for, imported lazily so this
    module imports with neither SDK present."""
    import httpx

    return (httpx.HTTPTransport, httpx.AsyncHTTPTransport)


def _is_standard_transport(transport: Any) -> bool:
    """EXACT TYPE IDENTITY. ``None`` counts as standard: an unset mount entry
    routes normally.

    Not a name comparison: ``type("HTTPTransport", (Rewriter,), {})`` is a
    body-rewriting transport that a name check certifies as standard. Not
    ``isinstance`` either: a genuine subclass overriding ``handle_request``
    passes that and can rewrite the body freely.

    The cost is deliberate. A benign third-party transport that subclasses
    ``HTTPTransport`` is reported ``unverified`` rather than ``covered``,
    which is the honest answer -- this agent cannot vouch for what an
    arbitrary transport does to the request body.
    """
    return transport is None or type(transport) in _standard_transports()


def client_escapes(client: Any) -> tuple[str, ...]:
    """Construction-time routes to the wire body that bypass inspection.

    A hostile application author is out of the threat model: anyone who can
    pass ``middleware=`` or ``transport=`` to the constructor can equally
    decline to install this agent. But out of the threat model is not
    unnoticed -- each of these is detected so the surface can be marked
    ``unverified`` rather than claiming a guarantee that does not hold.
    """
    found: list[str] = []
    if getattr(client, "_middleware", ()):
        found.append("middleware")

    inner = getattr(client, "_client", None)          # the httpx client
    if inner is not None:
        if not _is_standard_transport(getattr(inner, "_transport", None)):
            found.append("transport")

        # Mounts are inspected by transport TYPE, not by presence:
        # httpx.Client(proxy=...) populates _mounts with an ordinary
        # HTTPTransport, and flagging that would downgrade every application
        # behind a corporate proxy.
        for mounted in (getattr(inner, "_mounts", None) or {}).values():
            if not _is_standard_transport(mounted):
                found.append("mounts")
                break

        hooks = getattr(inner, "_event_hooks", {}) or {}
        if hooks.get("request"):
            found.append("event_hooks")
    return tuple(found)


# -- the entries -----------------------------------------------------------

@dataclass(frozen=True)
class Surface:
    """One patched method, and everything claimed about it.

    NOT ``slots=True``: ``cached_property`` writes into ``__dict__``, and a
    slotted instance has none.
    """

    module: str
    attribute: str
    kind: str                                   # "sync" | "async"
    provider: str                               # "openai" | "anthropic"
    version_range: str
    provider_fields: tuple[str, ...]
    path_map: dict[str, str]
    non_prompt_bearing: tuple[str, ...]
    known_lossy: tuple[str, ...]
    lossless_shapes: dict[str, Callable[[Any], bool]]
    opaque_subtrees: tuple[str, ...]
    unset_sentinel_refs: tuple[tuple[str, str], ...]
    lossless_for: tuple[str, ...]
    transform_into: Callable[[dict, list], dict]
    synthetic_nodes: dict[str, str] = field(default_factory=dict)
    span_input: bool = False
    span_output: bool = False

    def __hash__(self) -> int:
        return hash((self.module, self.attribute))

    @functools.cached_property
    def unset_sentinels(self) -> tuple[type, ...]:
        """The imported CLASSES, resolved lazily.

        Stored as references so importing this module never imports an SDK;
        resolved to types because ``isinstance`` needs types and a tuple of
        strings raises ``TypeError``.
        """
        return tuple(
            getattr(importlib.import_module(module), attribute)
            for module, attribute in self.unset_sentinel_refs
        )


_OPENAI_SENTINELS = (("openai._types", "Omit"), ("openai._types", "NotGiven"))
_ANTHROPIC_SENTINELS = (("anthropic._types", "NotGiven"), ("anthropic._types", "Omit"))

_OPENAI_PATH_MAP = {
    "messages":                                     "guard_input.messages",
    "messages[*]":                                  "guard_input.messages[*]",
    "messages[*].role":                             "guard_input.messages[*].role",
    "messages[*].content":                          "guard_input.messages[*].content",
    # NO `messages[*].tool_calls` ENTRIES, deliberately.
    #
    # `normalize_openai_messages` emits only `role` and `content`, and the
    # guard server reads only `content` (`app/routes/guard.py` joins
    # `m.get("content", "")`). Mapping tool calls it ignores would MOVE the
    # defect rather than fix it: the path would look mapped, the payload
    # would look inspected, and nothing would examine it.
    #
    # The consequence is fail-closed. An unmapped path is LOSSY, so a message
    # carrying `tool_calls` is refused in `enforce` before either the guard or
    # the provider is contacted, and in `monitor` it proceeds with a recorded
    # `lossy` skip and `is_active()` False. The agent declines to certify what
    # it cannot show the guard.
    "tools":                                        "guard_input.tools",
    "tools[*]":                                     "guard_input.tools[*]",
    "tools[*].function":                            "guard_input.tools[*].function",
    "tools[*].function.name":                       "guard_input.tools[*].function.name",
    "tools[*].function.description":                "guard_input.tools[*].function.description",
    "tools[*].function.parameters":                 "guard_input.tools[*].function.parameters",
}

_OPENAI_NON_PROMPT_BEARING = (
    "tools[*].type",        # the "function" discriminator; no guard equivalent
    # tool_choice and response_format are NOT mapped: the route reads
    # guard_input.messages and guard_input.tools and nothing else, so mapping
    # them would certify an inspection that does not happen. Both carry
    # predicates above for the variants that DO hold author text.
    "tool_choice", "response_format",
    "function_call",        # legacy control; its `functions` are known_lossy
    "model", "stream", "temperature", "top_p", "n", "max_tokens",
    "presence_penalty", "frequency_penalty", "logit_bias", "user",
    "seed", "stop", "stream_options", "parallel_tool_calls",
    "logprobs", "top_logprobs", "service_tier", "store", "metadata",
    "extra_headers", "extra_query", "extra_body", "timeout",
    "audio",                # output voice/format
    "modalities", "web_search_options", "max_completion_tokens",
    "reasoning_effort", "verbosity", "prompt_cache_key", "safety_identifier",
)

_OPENAI_SHAPES = {
    "messages[*].content": _is_str,
    "response_format":     _is_schemaless_response_format,
    "tool_choice":         _is_toolless_tool_choice,
    "extra_body":          _is_empty_extra_body,
    "extra_query":         _extra_query_cannot_carry_a_prompt,
}

OPENAI_CHAT_SYNC = Surface(
    module="openai.resources.chat.completions.completions",
    attribute="Completions.create",
    kind="sync",
    provider="openai",
    # The range CI actually tests. Eleven of the declared parameters do not
    # exist below ~1.40, so a wider floor would be a claim about versions
    # nothing has verified.
    version_range=">=1.40.0,<2.0.0",
    provider_fields=("messages", "tools", "tool_choice", "response_format"),
    path_map=_OPENAI_PATH_MAP,
    non_prompt_bearing=_OPENAI_NON_PROMPT_BEARING,
    # Known, prompt-bearing, and not sendable. Declaring these controls would
    # be a false certification; leaving them out relies on the unknown-key
    # fallback, which is weaker because these keys ARE known.
    known_lossy=("functions", "prediction"),
    lossless_shapes=_OPENAI_SHAPES,
    opaque_subtrees=("tools[*].function.parameters",),
    unset_sentinel_refs=_OPENAI_SENTINELS,
    # NO "tool calls". The executable contract refuses them: the paths are
    # unmapped, therefore lossy, so `enforce` declines the call rather than
    # showing the guard a body without them. Leaving the prose claim beside
    # the map that no longer backs it is how a reader learns the wrong
    # contract from the friendlier of the two.
    lossless_for=("string content", "tool definitions"),
    transform_into=_openai_write_back,
)

OPENAI_CHAT_ASYNC = Surface(
    **{**{f.name: getattr(OPENAI_CHAT_SYNC, f.name)
          for f in OPENAI_CHAT_SYNC.__dataclass_fields__.values()},
       "attribute": "AsyncCompletions.create", "kind": "async"}
)

_ANTHROPIC_PATH_MAP = {
    # `system` is a separate kwarg provider-side and becomes a SYNTHETIC first
    # message in guard input. `[1:]` is an INDEX PREDICATE ("any index >= 1"),
    # not a slice followed by another subscript: container pairs with
    # container, element with element.
    "system":                "guard_input.messages[0].content",
    "messages":              "guard_input.messages",
    "messages[*]":           "guard_input.messages[1:]",
    "messages[*].role":      "guard_input.messages[1:].role",
    "messages[*].content":   "guard_input.messages[1:].content",
    "tools":                 "guard_input.tools",
    "tools[*]":              "guard_input.tools[*]",
    "tools[*].name":         "guard_input.tools[*].function.name",
    "tools[*].description":  "guard_input.tools[*].function.description",
    "tools[*].input_schema": "guard_input.tools[*].function.parameters",
}

ANTHROPIC_MESSAGES_SYNC = Surface(
    module="anthropic.resources.messages.messages",
    attribute="Messages.create",
    kind="sync",
    provider="anthropic",
    # `tools` does not reach the stable Messages.create until 0.27.0 -- it was
    # on the beta namespace before that, and 0.20-0.26 expose no `tools` at
    # all, which would make this entry's tools mapping dead code.
    version_range=">=0.27.0,<1.0.0",
    provider_fields=("messages", "system", "tools", "tool_choice"),
    path_map=_ANTHROPIC_PATH_MAP,
    non_prompt_bearing=(
        "tool_choice",      # same rationale as OpenAI; the route never reads it
        "model", "stream", "max_tokens", "temperature", "top_p", "top_k",
        "stop_sequences", "metadata", "extra_headers", "extra_query",
        "extra_body", "timeout",
        # Types opened and checked rather than classified from their names.
        "cache_control", "container", "inference_geo", "output_config",
        "service_tier", "thinking", "user_profile_id",
    ),
    known_lossy=(),         # no legacy tool API on this surface
    lossless_shapes={
        "messages[*].content": _is_str,
        "system":              _is_str,
        "output_config":       _is_schemaless_output_config,
        "extra_body":          _is_empty_extra_body,
        "extra_query":         _extra_query_cannot_carry_a_prompt,
    },
    opaque_subtrees=("tools[*].input_schema",),
    unset_sentinel_refs=_ANTHROPIC_SENTINELS,
    lossless_for=("string content", "system prompt", "tool definitions"),
    transform_into=_anthropic_write_back,
    # Nodes the normalizer creates with no provider source. Declared, because
    # the bijection's second direction otherwise reports them as unsourced --
    # and an unsourced guard node IS normally a normalizer bug.
    synthetic_nodes={
        "guard_input.messages[0]":       "system",
        "guard_input.messages[0].role":  "system",
        "guard_input.tools[*].function": "the flat->nested tool conversion",
    },
)

ANTHROPIC_MESSAGES_ASYNC = Surface(
    **{**{f.name: getattr(ANTHROPIC_MESSAGES_SYNC, f.name)
          for f in ANTHROPIC_MESSAGES_SYNC.__dataclass_fields__.values()},
       "attribute": "AsyncMessages.create", "kind": "async"}
)

SURFACES: tuple[Surface, ...] = (
    OPENAI_CHAT_SYNC, OPENAI_CHAT_ASYNC,
    ANTHROPIC_MESSAGES_SYNC, ANTHROPIC_MESSAGES_ASYNC,
)


# -- resolution and classification ----------------------------------------

def resolve(surface: Surface) -> Callable[..., Any]:
    """The UNBOUND function off the SDK class, e.g. ``Completions.create``."""
    module = importlib.import_module(surface.module)
    class_name, method_name = surface.attribute.split(".")
    return getattr(getattr(module, class_name), method_name)


def _shape_for(surface: Surface, path: str):
    """The first declared predicate whose PATTERN matches this CONCRETE path.

    Pattern matching, not dict lookup: ``bound_nodes`` yields
    ``messages[0].content`` and the rule is keyed ``messages[*].content``.
    """
    for pattern, predicate in surface.lossless_shapes.items():
        if matches(pattern, path):
            return predicate
    return None


def _is_mapped(surface: Surface, path: str) -> bool:
    return any(matches(source, path) for source in surface.path_map)


def _is_non_prompt_bearing(surface: Surface, path: str) -> bool:
    """Pattern matching, not membership: entries such as ``tools[*].type``
    never equal a concrete path."""
    return any(matches(pattern, path) for pattern in surface.non_prompt_bearing)


#: Types whose value the agent can compare and the provider will serialise
#: predictably. EXACT types only: a subclass can override `__eq__`, `__str__`
#: or `__reduce__` and make the value it compares as differ from the value it
#: serialises as.
_EXACT_SCALARS = (str, int, float, bool)


_MEMBER_DESCRIPTOR = type(type("_S", (), {"__slots__": ("x",)}).x)

# The walk is bounded, because it is recursive. A deeply nested tool schema
# or a self-referencing container would otherwise raise `RecursionError`, and
# `content_fingerprint` is called OUTSIDE dispatch's try block -- so it would
# escape as a crash attributable to this agent rather than as a verdict.
# Exhausting a budget is not a licence to wave the value through: it
# snapshots as unvouchable, which is what the agent actually knows.
#
# Sized from measurement, not guesswork. A 100-message conversation is ~500
# nodes and 0.2ms; tool schemas dominate, and 100 tools of 100 fields each is
# ~61k nodes and 14ms. 250k covers 200x200 at ~70ms, which the guard's own
# network round-trip dwarfs. Exceeding it is not a licence to proceed: an
# incomplete fingerprint refuses in enforce, because a constant sentinel
# cannot show a change beneath it.
_MAX_DEPTH = 50
_MAX_NODES = 250_000


def _slot_state(value: Any, depth: int, path: frozenset, budget: list) -> tuple:
    """Values stored in `__slots__`, read through the member descriptor.

    `__dict__` is not all of an object's state. Pydantic v2 keeps declared
    fields there, but an `extra="allow"` model keeps unknown fields in
    `__pydantic_extra__` -- a slot -- and `model_dump()` serialises them. An
    extra mutated during the guard call therefore changed the wire payload
    while a `__dict__`-only fingerprint stayed byte-identical.

    Pydantic is one instance of the general defect, so this is general:
    every slot declared anywhere in the MRO. Reading through the descriptor
    rather than attribute access is the same rule the mapping branch follows
    with `dict.items` -- an overridden `__getattribute__` must not be able to
    dress up what is actually stored.
    """
    items = []
    for klass in type(value).__mro__:
        declared = klass.__dict__.get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        for name in declared:
            if name in ("__dict__", "__weakref__"):
                continue
            descriptor = klass.__dict__.get(name)
            if not isinstance(descriptor, _MEMBER_DESCRIPTOR):
                continue                    # shadowed by a class attribute
            try:
                items.append((str(name), _trusted(
                    descriptor.__get__(value, klass), depth + 1, path, budget)))
            except AttributeError:
                items.append((str(name), ("unset", None)))
            except Exception:               # pragma: no cover - defensive
                items.append((str(name), ("?", "unreadable")))
    return tuple(sorted(items, key=repr))


def _trusted(value: Any, depth: int = 0, path: frozenset = frozenset(),
             budget: list | None = None) -> Any:
    """A snapshot comparable with BUILT-IN operations only.

    Never `!=`, which calls the value's own `__eq__`: a `str` subclass can
    store an attack and report itself equal to a benign string, so the guard
    sees the benign reading while the provider serialises the attack.
    Equality supplied by the thing under inspection cannot be the boundary.

    Types are tagged and exact, so `LyingStr("attack")` snapshots as
    `("?", "LyingStr")` rather than as a string, and nothing it overrides can
    make that match `("str", "benign text")`.
    """
    if budget is None:
        budget = [_MAX_NODES]
    budget[0] -= 1
    if budget[0] < 0:
        return ("?", "budget-exhausted")
    if depth > _MAX_DEPTH:
        return ("?", "too-deep")

    if value is None:
        return ("none", None)
    if type(value) in _EXACT_SCALARS:
        return (type(value).__name__, value)

    # Containers can refer to themselves. Track identity along THIS path
    # only, so a value shared between siblings still snapshots normally.
    if id(value) in path:
        return ("?", "cycle")
    inner = path | {id(value)}

    if type(value) in (list, tuple):
        return ("list", tuple(_trusted(item, depth + 1, inner, budget)
                              for item in value))
    if type(value) in (set, frozenset):
        # Ordinary data, and reachable through slots: pydantic keeps
        # `__pydantic_fields_set__` as a set. Leaving it to the untrusted
        # fallback would mark every pydantic-bearing message divergent and
        # refuse legitimate traffic in enforce. Sorted by snapshot, because
        # iteration order is not part of the value.
        return ("set", tuple(sorted(
            (_trusted(item, depth + 1, inner, budget) for item in value),
            key=repr)))

    if isinstance(value, dict):
        # SUBCLASSES INCLUDED, read through `dict.items` rather than the
        # instance's own method. Snapshotting a subclass by type name alone
        # leaves in-place mutation invisible -- the type does not change, so
        # before and after match and content swapped during the guard call
        # reaches the provider uninspected. Reading the stored items is also
        # what makes the snapshot immune to an overridden `items()` or
        # `get()`.
        return ("dict", type(value).__name__, tuple(sorted(
            (str(key), _trusted(item, depth + 1, inner, budget))
            for key, item in dict.items(value))))

    # SCALAR SUBCLASSES ARE NEVER TRUSTED, and this must come before the
    # object branch below. A `str` subclass has a `__dict__` like any other
    # object, so the object branch would snapshot `LyingStr("MALICIOUS")` as
    # ("obj", "LyingStr", ()) -- stable, comparable and entirely content-free,
    # which re-opens the bypass the exact-type rule closes. A value that
    # serialises as text must be exactly text.
    if isinstance(value, (str, bytes, int, float, bool)):
        return ("?", type(value).__name__)

    # Objects with instance state -- pydantic message and tool models are the
    # ordinary case, and the SDKs hand them to callers routinely. Snapshot
    # their `__dict__` through `object.__getattribute__`, so a descriptor or
    # `__getattr__` cannot dress up what is actually stored.
    try:
        state = object.__getattribute__(value, "__dict__")
    except Exception:
        state = None
    slots = _slot_state(value, depth, inner, budget)
    if type(state) is dict:
        return ("obj", type(value).__name__, tuple(sorted(
            (str(key), _trusted(item, depth + 1, inner, budget))
            for key, item in dict.items(state))), slots)
    if slots:
        return ("obj", type(value).__name__, (), slots)

    # Anything else is not something this agent can vouch for.
    return ("?", type(value).__name__)


def fingerprint_is_incomplete(snapshot: Any) -> bool:
    """Whether a snapshot OMITS content, so a change beneath it is invisible.

    Distinct from `_is_untrusted`, which also covers values that were fully
    read but cannot be vouched for. This asks the narrower question the
    mutation check needs: did the walk stop early? Two identical sentinels
    compare equal, so anything they replaced could have changed silently.
    """
    if isinstance(snapshot, tuple):
        if snapshot[:1] == ("?",) and snapshot[1:] in (
                ("too-deep",), ("budget-exhausted",), ("cycle",), ("unreadable",)):
            return True
        return any(fingerprint_is_incomplete(part) for part in snapshot)
    return False


def _is_untrusted(snapshot: Any) -> bool:
    """Whether a snapshot contains a value the agent cannot vouch for."""
    if isinstance(snapshot, tuple) and snapshot and snapshot[0] == "?":
        return True
    if isinstance(snapshot, tuple):
        return any(_is_untrusted(part) for part in snapshot)
    return False


def content_fingerprint(surface: "Surface", kwargs: dict,
                        args: tuple = ()) -> Any:
    """A trusted snapshot of everything the guard was shown.

    Classification happens BEFORE the guard call and the provider is invoked
    with the caller's own mutable kwargs AFTER it. Anything that runs in
    between -- another thread, a callback, a re-entrant guard -- can replace
    message content once inspection is finished, and the provider then
    receives text nothing examined. The async path holds that window open for
    the whole await.

    Comparing fingerprints across the guard call closes it: not by locking
    the caller's data, which this agent does not own, but by refusing to
    proceed when what it inspected is no longer what it would send.

    EVERY argument, not just the guarded fields. Anything able to change what
    the provider sends must be inside the comparison -- including arguments
    whose CLASSIFICATION changes when they are mutated. A non-empty
    `extra_body` REPLACES the provider's wire body, so a shared empty
    `extra_body={}` passes classification as lossless precisely because it is
    empty, and can then be filled with an entire replacement conversation
    before the provider is invoked. Narrowing this to `provider_fields` puts
    it outside the comparison and both fingerprints match, because neither
    looked.

    Positional arguments too. Every declared surface is keyword-only in the
    SDK versions in front of us, so `args` is empty in practice -- but that is
    an observation, not a guarantee, and a parameter that became positional
    would otherwise sit silently outside the comparison.
    `test_every_declared_surface_is_KEYWORD_ONLY` says so if that changes.
    """
    return (tuple(_trusted(value) for value in args),
            tuple(sorted((str(field), _trusted(value))
                         for field, value in kwargs.items())))


def _divergent_containers(nodes: dict[str, Any]) -> list[str]:
    """Paths whose container reads differently through `get()` than it stores.

    Only mappings are checked, and only where the two readings actually
    differ: the point is not to reject subclasses, which are ordinary in SDK
    code, but to refuse the specific case where the agent cannot tell which
    value the provider will serialise.
    """
    divergent: list[str] = []
    for path, value in nodes.items():
        if not isinstance(value, dict):
            continue
        for key in list(value.keys()):
            try:
                stored = dict.__getitem__(value, key)
                read = value.get(key)
            except Exception:               # pragma: no cover - defensive
                divergent.append(f"{path}.{key}")
                continue
            # Compare SNAPSHOTS, never the values themselves: `!=` calls
            # the value's own `__eq__`, which the caller controls.
            if _trusted(stored) != _trusted(read):
                divergent.append(f"{path}.{key}")
            elif _is_untrusted(_trusted(stored)):
                # Both readings agree and neither can be vouched for -- a
                # scalar subclass can serialise differently from how it
                # compares, so the agent cannot say what the provider sends.
                divergent.append(f"{path}.{key}")
    return divergent


def lossy_paths(surface: Surface, nodes: dict[str, Any]) -> tuple[str, ...]:
    """Every bound path this surface cannot faithfully send.

    Three sources, all fail-closed:

    1. a path matching a ``known_lossy`` pattern -- known, prompt-bearing and
       not sendable;
    2. a declared shape predicate returning False;
    3. a path in neither ``path_map`` nor ``non_prompt_bearing`` -- unknown.

    ``nodes`` must already have had unsupplied parameters dropped by
    ``bound_nodes``: without that, (3) fires for every sentinel-defaulted
    parameter and enforce mode refuses everything.
    """
    lossy: list[str] = []

    # FOURTH source, and the only one about the container rather than the
    # path: a mapping whose `get()` disagrees with what it stores.
    #
    # Classification walks values one way and the normalizer reads them
    # another -- `items()`/`vars()` here, `get()`/`getattr()` there. A dict
    # SUBCLASS that overrides `get("content")` therefore showed the guard
    # "benign text" while the provider serialised the stored value, and the
    # call proceeded in enforce: one payload inspected, a different one sent,
    # with nothing in between noticing.
    #
    # Fail-closed rather than clever: if the two readings of a container
    # disagree, the agent cannot say which one the provider will use, so the
    # call is lossy and enforce declines it. A plain dict, and any container
    # whose readings agree, is unaffected.
    lossy.extend(_divergent_containers(nodes))

    for path, value in nodes.items():
        if any(matches(pattern, path) for pattern in surface.known_lossy):
            lossy.append(path)
            continue

        predicate = _shape_for(surface, path)
        if predicate is not None and not predicate(value):
            lossy.append(path)
            continue

        if not _is_mapped(surface, path) and not _is_non_prompt_bearing(surface, path):
            lossy.append(path)
    return tuple(lossy)


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in version.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def disposition_for(surface: Surface, installed_version: str) -> str:
    """``covered`` when the installed SDK is inside the tested range, else
    ``unverified``. Computed at activation, never stored on the entry."""
    installed = _parse_version(installed_version)
    for clause in surface.version_range.split(","):
        clause = clause.strip()
        if clause.startswith(">="):
            if installed < _parse_version(clause[2:]):
                return "unverified"
        elif clause.startswith("<"):
            if installed >= _parse_version(clause[1:]):
                return "unverified"
    return "covered"
