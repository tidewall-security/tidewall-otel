"""Dispatch-time coverage: is this call one we can honestly guard?

Coverage is decided per call, not per patch. A boundary being patched says
only that we are on the call path; whether the *particular* arguments can be
faithfully represented to the guard is a separate question, and answering it
wrongly is how an agent reports enforcement while sending uninspected content.

CANONICALISATION
----------------

The structural equivalence between what the provider was asked to do and what
the guard was shown needs one definition of "the same node", stated exhaustively
so two implementations cannot disagree:

=========================  =================================================
construct                  rule
=========================  =================================================
object                     key ORDER is not significant
array                      order IS significant -- it changes who said what
string                     compared exactly, including whitespace
number, bool, null         compared by TYPE and value: 1, 1.0 and "1" are
                           three different nodes, and True is not 1
binary / media reference   compared by its reference form, never by decoded
                           content
=========================  =================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._manifest import Surface, lossy_paths


def canonicalise(value: Any) -> Any:
    """A hashable form in which equal nodes compare equal, and nothing else does.

    Types are carried explicitly because Python's own equality is too
    generous for this purpose: ``True == 1`` and ``1 == 1.0`` are both true,
    and a guard that inherits that cannot tell a boolean flag from a count, or
    a numeric field from a string of digits.
    """
    if isinstance(value, dict):
        # Sorted by key: object key order is not significant.
        return ("object", tuple(sorted(
            (key, canonicalise(item)) for key, item in value.items()
        )))
    if isinstance(value, (list, tuple)):
        # Positional: array order IS significant.
        return ("array", tuple(canonicalise(item) for item in value))
    if isinstance(value, (bytes, bytearray)):
        # By reference form. Decoding to compare would mean normalising
        # attacker-controlled binary in memory.
        return ("binary", bytes(value))
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        # BEFORE int: bool is a subclass of int, and True would otherwise
        # canonicalise identically to 1.
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("string", value)
    return ("opaque", repr(value))


@dataclass(frozen=True)
class Coverage:
    """The verdict for one bound call."""

    surface: Surface
    lossy_paths: tuple[str, ...]

    @property
    def is_lossless(self) -> bool:
        return not self.lossy_paths


def classify_input(surface: Surface, nodes: dict[str, Any]) -> Coverage:
    """Classify a bound call against its surface's contract.

    Delegates to the manifest's :func:`lossy_paths` rather than re-deriving
    lossiness here. The predicates live beside the declarations they evaluate,
    so there is exactly one definition of what lossy means and dispatch cannot
    drift from it.
    """
    return Coverage(surface=surface, lossy_paths=lossy_paths(surface, nodes))
