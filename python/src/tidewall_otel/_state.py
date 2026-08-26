"""Agent state as INDEPENDENT dimensions (spec section 4).

Four separate facts, deliberately not collapsed into one boolean:

``lifecycle``   uninstalled / installing / installed / removed / residual
``mode``        enforce / monitor / dry-run
``surfaces``    per-surface disposition: covered / unverified / uncovered /
                refusing
``guard_health`` unknown / ok / degraded / or the failing outcome kind
                (`unreachable`, `timeout`, `saturated`, `schema_invalid`,
                `invariant_violated`)

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
#: Outcomes that mean the guard answered and this agent understood it. Every
#: other kind is reported under its own name.
_GUARD_HEALTH = {"clean": "ok", "blocked": "ok", "transformed": "ok"}
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
    #: Keys of events already recorded. An operator needs to know WHICH
    #: conditions occurred, not how many times: a process using a custom
    #: httpx client recorded one `client_escapes` event per LLM call and grew
    #: this list forever, and monitor mode did the same with one `skip` per
    #: lossy call. Both are steady-state conditions that repeat on every call
    #: by their nature, so the log they produced was unbounded by design.
    _recorded: set = field(default_factory=set, repr=False, compare=False)

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
        self._record_once(StateEvent("unverified", surface, reason, detail))

    def record_guard_health(self, outcome_kind: str) -> None:
        """What the last guard call says about the guard.

        The dimension existed, was documented, and was tested by direct
        construction -- and no production code ever wrote to it. An operator
        polling `state()` through a total guard outage saw `unknown` from
        activation onwards while every enforce-mode call failed.

        A SCALAR, deliberately: recording an event per failed call would grow
        without bound during exactly the outage an operator most needs to
        survive. The current value answers "is the guard working", which is
        what the dimension is for; the failing calls raise, and their spans
        carry the per-call detail.

        The vocabulary is the outcome kind itself rather than a flattened
        `unreachable`, because `saturated` is this agent's own pool declining
        work and `schema_invalid` is the guard answering badly -- calling
        either "unreachable" would misdirect whoever is paging.
        """
        self.guard_health = _GUARD_HEALTH.get(outcome_kind, outcome_kind)

    def record_history(self, subject: str, reason: str, detail: Any = None) -> None:
        """Record something that HAPPENED, without claiming anything current.

        `record_unverified` downgrades a surface, which is right for a live
        boundary the agent cannot vouch for and wrong for a fact about a past
        one. Permanent residuals are history: the class they name has been
        collected, so it is not a boundary any more -- and a NEW class
        imported under the same module path is a different object that may
        well have been patched and removed cleanly. Replaying the old record
        as a downgrade marked that replacement `unverified` forever.

        So: an event, never a disposition.
        """
        self.events.append(StateEvent("unrecoverable", subject, reason, detail))

    def record_skip(self, surface: str, reason: str, detail: Any = None) -> None:
        """Record a guard call that was deliberately not made.

        Monitor mode exists to produce evidence, so the one call worth
        recording must not be the one that records nothing.
        """
        self._record_once(StateEvent("skip", surface, reason, detail))

    def _record_once(self, event: StateEvent) -> None:
        """Append an event the first time its exact condition occurs.

        Deduplicated on every field, so a NEW reason or a new detail is still
        recorded -- what is dropped is the hundredth identical report of a
        condition already visible in the log.
        """
        key = (event.kind, event.surface, event.reason, repr(event.detail))
        if key in self._recorded:
            return
        self._recorded.add(key)
        self.events.append(event)

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
