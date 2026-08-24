"""Agent state as INDEPENDENT dimensions (spec section 4).

Four separate facts, deliberately not collapsed into one boolean:

``lifecycle``   uninstalled / installing / installed / removed / residual
``mode``        enforce / monitor / dry-run
``surfaces``    per-surface disposition: covered / unverified / uncovered /
                refusing
``guard_health`` unknown / ok / unreachable

``residual`` is deactivation that could not fully undo itself: another agent
wrapped a boundary after us, so removal correctly declined to write and a
Tidewall wrapper is still live underneath theirs. Reporting that as ``removed``
would tell an operator the SDK is pristine while our code runs on every call.

Collapsing them is how an agent ends up reporting ``active`` while a boundary
is unguarded: a single flag has to pick one fact to represent, and every choice
loses something a caller needs.

``is_active()`` is a UNIVERSAL claim over present surfaces, and deliberately
false when there are none. With no SDK imported, "every present surface is
covered" is vacuously true -- so an agent guarding nothing would otherwise
report itself enforcing, which is the most misleading answer available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_COVERED = "covered"
_ENFORCING_MODES = frozenset({"enforce", "monitor"})


@dataclass
class StateEvent:
    """Something worth telling an operator about, with its reason."""

    kind: str                       # "skip" | "unverified"
    surface: str
    reason: str
    detail: Any = None

    def __str__(self) -> str:
        return f"{self.kind} {self.surface}: {self.reason} {self.detail}"


@dataclass
class State:
    """The agent's own account of itself."""

    lifecycle: str = "uninstalled"
    mode: str = "enforce"
    surfaces: dict[str, str] = field(default_factory=dict)
    #: `unknown`, not `ok`. Preflight is optional, so there is no evidence
    #: either way until the first call, and `ok` would be a claim with nothing
    #: behind it.
    guard_health: str = "unknown"
    events: list[StateEvent] = field(default_factory=list)

    def is_active(self) -> bool:
        """Whether the agent is enforcing across every boundary present.

        Universal rather than existential: one uncovered surface makes the
        claim false, because a caller cannot know which boundary they crossed.

        Guard health is deliberately NOT part of this. An unreachable guard is
        a runtime condition the mode contract handles per call; it does not
        retroactively mean the boundaries are unguarded.
        """
        if self.lifecycle != "installed":
            return False
        if self.mode not in _ENFORCING_MODES:
            return False            # dry-run skips guard calls entirely
        if not self.surfaces:
            return False            # the vacuous-true case
        return all(disposition == _COVERED for disposition in self.surfaces.values())

    def record_unverified(self, surface: str, reason: str, detail: Any = None) -> None:
        """Downgrade a surface and say why.

        Out of the threat model is not the same as unnoticed: when the agent
        cannot vouch for a boundary it must stop claiming it, and an operator
        needs the reason to decide whether they care.
        """
        self.surfaces[surface] = "unverified"
        self.events.append(StateEvent("unverified", surface, reason, detail))

    def record_skip(self, surface: str, reason: str, detail: Any = None) -> None:
        """Record a guard call that was deliberately not made.

        Monitor mode exists to produce evidence, so the one call worth
        recording must not be the one that records nothing.
        """
        self.events.append(StateEvent("skip", surface, reason, detail))

    def summary(self) -> str:
        """One line naming every dimension.

        ``active: false`` without a reason is not actionable, so the reason is
        always present.
        """
        dispositions = ", ".join(
            f"{name}={disposition}" for name, disposition in sorted(self.surfaces.items())
        ) or "no supported surface present"
        return (
            f"tidewall: active={self.is_active()} "
            f"lifecycle={self.lifecycle} mode={self.mode} "
            f"guard={self.guard_health} surfaces[{dispositions}]"
        )
