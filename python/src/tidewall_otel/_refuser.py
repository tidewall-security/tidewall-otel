"""Refusers: what gets installed when a boundary cannot be trusted.

When activation cannot establish that a surface is covered -- an unverified
SDK version, a guard that will not answer, coverage that could not be proved
-- the honest response in ``enforce`` is to stop calls reaching the provider
rather than to let them through unguarded while reporting enforcement.

A refuser is installed in place of the wrapper and raises. It never calls
through: that is the whole point, and it is what separates "refused" from
"failed open".

SHAPE MATTERS. An async surface needs an awaitable refuser. Installing a
synchronous one on an async method makes ``await`` raise ``TypeError`` rather
than the refusal -- the provider is still not called, so a test asserting only
that "something raised" cannot tell the two apart, while an operator sees an
error that says nothing about why.
"""

from __future__ import annotations

from typing import Any, Callable

from ._exceptions import TidewallError
from ._manifest import Surface

#: Marks a callable as one of ours. The patch manager may remove its own
#: refusers and must never remove a foreign wrapper, so it needs to tell them
#: apart by more than a guess.
_REFUSER_MARKER = "__tidewall_refuser__"


class TidewallActivationRefusedError(TidewallError):
    """Raised in place of a provider call the agent cannot vouch for.

    A `TidewallError`, NOT a bare `RuntimeError`. Under
    ``TIDEWALL_ON_ACTIVATION_FAILURE=block`` this is raised for EVERY call, so
    an application that carefully wraps its AI calls in `except TidewallError`
    would have missed all of them and crashed on an exception it had no reason
    to expect -- the failure mode the block policy exists to prevent, arriving
    in a shape the caller cannot handle.
    """


def _message(surface: Surface, reason: str) -> str:
    return (
        f"Tidewall refused {surface.attribute} on {surface.module}: {reason}. "
        f"The call did not reach the provider."
    )


def make_refuser(surface: Surface, reason: str) -> Callable[..., Any]:
    """Build a refuser matching this surface's calling convention.

    The returned callable takes wrapt's ``(wrapped, instance, args, kwargs)``
    so it can be installed exactly where a wrapper would be.
    """
    if surface.kind == "async":
        async def refuser(wrapped=None, instance=None, args=(), kwargs=None):
            raise TidewallActivationRefusedError(_message(surface, reason))
    else:
        def refuser(wrapped=None, instance=None, args=(), kwargs=None):
            raise TidewallActivationRefusedError(_message(surface, reason))

    setattr(refuser, _REFUSER_MARKER, True)
    refuser.tidewall_surface = surface
    refuser.tidewall_reason = reason
    return refuser


def is_refuser(obj: Any) -> bool:
    """Whether ``obj`` is one of our refusers.

    Attribute-marked rather than inferred: the patch manager decides what it
    may safely remove on this answer, and guessing from a name or a module
    would let it remove another agent's wrapper.
    """
    return getattr(obj, _REFUSER_MARKER, False) is True
