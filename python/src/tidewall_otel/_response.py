"""The guard response schema, and a TOTAL classifier (spec section 11).

Total means every well-formed input produces exactly one verdict. A classifier
with gaps returns nothing for some combination, and dispatch's catch-all then
files a legitimate guard verdict as ``invariant_violated`` -- turning a policy
decision into an apparent bug, in the direction that fails open or refuses
arbitrarily depending on mode.

The required-field set is PINNED against the server package rather than copied.
A copy drifts silently; a pin fails loudly when the server changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Required by the server's GuardResult. Pinned by a test that imports the
#: server model when it is importable, and records the skip when it is not --
#: a pin that quietly does not run is not a pin.
REQUIRED_RESULT_FIELDS = frozenset({"blocked", "transformed", "policy"})

#: Optional, with the server's own defaults. Rejecting a missing one would
#: refuse valid responses.
OPTIONAL_RESULT_DEFAULTS: dict[str, Any] = {
    "guard_output": None,
    "detectors": dict,
    "access_rules": dict,
    "fpe_context": None,
    "degraded": False,
    "failed_detectors": list,
}

_TYPES: dict[str, type | tuple[type, ...]] = {
    "blocked": bool,
    "transformed": bool,
    "policy": str,
    "degraded": bool,
    "failed_detectors": list,
    "detectors": dict,
    "access_rules": dict,
}


@dataclass(frozen=True)
class Outcome:
    """One verdict about one guard response."""

    kind: str                       # blocked | transformed | degraded | clean
                                    # | schema_invalid | unreachable | timeout
                                    # | saturated | invariant_violated
    guard_output: dict | None = None
    failed_detectors: tuple[str, ...] = ()
    detail: str = ""
    raw: dict = field(default_factory=dict)


def _invalid(detail: str) -> Outcome:
    return Outcome(kind="schema_invalid", detail=detail)


def _usable_transform(result: dict) -> dict | None:
    """A transform payload we can actually apply, or None.

    ``guard_output`` must carry messages: an empty or shapeless object cannot
    be written back, and treating it as applicable would send the caller's
    original prompt while the guard believed it had rewritten it.
    """
    guard_output = result.get("guard_output")
    if isinstance(guard_output, dict) and guard_output.get("messages"):
        return guard_output
    return None


def classify_response(payload: Any) -> Outcome:
    """Classify a decoded guard response. Total over well-formed input."""
    if not isinstance(payload, dict) or not payload:
        return _invalid(f"response body is not a non-empty object: {type(payload).__name__}")

    result = payload.get("result")
    if not isinstance(result, dict):
        return _invalid("response has no `result` object")

    missing = REQUIRED_RESULT_FIELDS - set(result)
    if missing:
        return _invalid(f"missing required field(s): {sorted(missing)}")

    # Type-check required fields and any PRESENT optional one. Unknown fields
    # are accepted at both levels: both server models declare extra="allow",
    # so refusing them would break on a server minor release.
    # NOTE: no separate "bool masquerading as int" guard. bool subclasses int,
    # so one would be needed if any field were typed int or float -- none is,
    # every expected type here is bool, str, list or dict, and isinstance(True,
    # str) is already False. Mutation-testing showed such a guard could never
    # execute: dead code implying a protection that is not operating. Add it
    # back together with the first numeric field, and with a test.
    for name, expected in _TYPES.items():
        if name in result and not isinstance(result[name], expected):
            return _invalid(f"field {name!r} has type {type(result[name]).__name__}")

    failed = tuple(result.get("failed_detectors") or ())

    # BLOCKED WINS. An unusable transform payload does not turn a refusal into
    # a bug: the safe reading of an ambiguous verdict is the more restrictive
    # one, and the caller asked to be protected.
    if result["blocked"]:
        return Outcome(kind="blocked", failed_detectors=failed, raw=payload)

    if result["transformed"]:
        guard_output = _usable_transform(result)
        if guard_output is None:
            # NOT clean: reporting clean would let the unmodified prompt
            # proceed while the guard believed it had rewritten it.
            return _invalid("transformed=true with no usable guard_output")
        return Outcome(kind="transformed", guard_output=guard_output,
                       failed_detectors=failed, raw=payload)

    if result.get("degraded"):
        # Distinct from clean: "checked, found nothing" and "could not check"
        # are different answers, and a caller that cannot tell them apart
        # cannot act on either.
        return Outcome(kind="degraded", failed_detectors=failed, raw=payload)

    return Outcome(kind="clean", failed_detectors=failed, raw=payload)
