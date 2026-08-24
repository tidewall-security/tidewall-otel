"""The bound-call primitive: what the provider was actually asked to do.

Coverage is defined over the arguments a call is bound to, so this module
turns ``(callable, args, kwargs)`` into a flat mapping of path to value. Every
downstream predicate keys on the paths produced here, so the grammar is part
of the contract rather than an implementation detail.

THE PATH GRAMMAR
----------------

A path is built from the bound parameter name outward:

===================================  ==========  ==========================
step                                 syntax      example
===================================  ==========  ==========================
bound parameter                      the name    ``messages``
mapping key **or public attribute**  ``.name``   ``messages[0].content``
sequence element                     ``[i]``     ``messages[0]``
===================================  ==========  ==========================

Mappings and objects are walked the same way. SDK *input* types are TypedDicts
-- plain dicts at runtime -- but a *response* object fed back into the next
request is a pydantic model with no ``.get``, and a walk written for mappings
alone would see nothing inside it.

Patterns add exactly two wildcards and nothing else: ``[*]`` matches any
``[i]``, and ``[n:]`` matches any ``[i]`` with ``i >= n``.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Iterable, Mapping

_SEG = re.compile(r"[^.\[]+|\[\*\]|\[\d+:\]|\[\d+\]")


def matches(pattern: str, path: str) -> bool:
    """Segment-wise comparison of a pattern against a concrete path.

    ``[*]``  matches any ``[i]``.
    ``[n:]`` matches any ``[i]`` with ``i >= n`` (the synthetic-head case).
    Every other segment must be identical.
    """
    pattern_segments = _SEG.findall(pattern)
    path_segments = _SEG.findall(path)
    if len(pattern_segments) != len(path_segments):
        return False

    for expected, actual in zip(pattern_segments, path_segments):
        if expected == actual:
            continue
        if not (actual.startswith("[") and actual.endswith("]")
                and actual[1:-1].isdigit()):
            return False
        if expected == "[*]":
            continue
        if (expected.endswith(":]") and expected[1:-2].isdigit()
                and int(actual[1:-1]) >= int(expected[1:-2])):
            continue
        return False
    return True


def _is_unset(value: Any, unset: tuple[type, ...]) -> bool:
    """Whether a bound value is one of the SDK's not-supplied sentinels.

    Dropping these is load-bearing: a minimal real call binds 39 arguments on
    the OpenAI method, 33 of them sentinels. Walking them means any parameter
    not yet classified makes every call lossy -- the fail-closed rule inverted
    into refusing all traffic.
    """
    return bool(unset) and isinstance(value, unset)


def _walk(value: Any, prefix: str, stop: Iterable[str], out: dict[str, Any]) -> None:
    out[prefix] = value
    if any(matches(pattern, prefix) for pattern in stop):
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            _walk(item, f"{prefix}.{key}", stop, out)
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, f"{prefix}[{index}]", stop, out)
        return

    # Objects: walk public attributes. Anything with a __dict__ that is not a
    # str/bytes/scalar is treated as structured. Strings are leaves -- walking
    # their characters would be absurd and would also never terminate usefully.
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        for name, item in vars(value).items():
            if not name.startswith("_"):
                _walk(item, f"{prefix}.{name}", stop, out)


def bound_nodes(
    wrapped: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    *,
    stop: Iterable[str] = (),
    unset: tuple[type, ...] = (),
) -> dict[str, Any]:
    """Every node of a bound call, keyed by path.

    Binds against the callable's signature with ``apply_defaults()``, so a
    parameter left at its default is still visible: omitting it would hide a
    node from the equivalence.

    ``self`` is supplied to ``bind`` when the signature declares it -- the
    manifest resolves the unbound function off the class, and binding without
    a receiver raises ``TypeError`` -- and is never yielded, because an
    unaccounted node is lossy and ``self`` would make every call lossy.
    """
    signature = inspect.signature(wrapped)
    parameters = list(signature.parameters)

    bind_args = args
    if parameters and parameters[0] == "self" and len(args) < len(parameters):
        bind_args = (None, *args)

    bound = signature.bind(*bind_args, **kwargs)
    bound.apply_defaults()

    nodes: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name == "self":
            continue
        parameter = signature.parameters[name]
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            # **kwargs arrives as a dict of the extra names; each is a node in
            # its own right, not a node called "kwargs".
            for extra_name, extra_value in value.items():
                if not _is_unset(extra_value, unset):
                    _walk(extra_value, extra_name, stop, nodes)
            continue
        if _is_unset(value, unset):
            continue
        _walk(value, name, stop, nodes)
    return nodes
