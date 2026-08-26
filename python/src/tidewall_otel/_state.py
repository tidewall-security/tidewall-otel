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

import threading
from dataclasses import dataclass, field
from typing import Any

_COVERED = "covered"
#: Outcomes that mean the guard answered and this agent understood it. Every
#: other kind is reported under its own name.
_GUARD_HEALTH = {"clean": "ok", "blocked": "ok", "transformed": "ok"}
_ENFORCING_MODES = frozenset({"enforce", "monitor"})


#: Distinct details kept per condition. An operator needs examples, not a
#: transcript: the details are caller-controlled and can grow per call.
_MAX_SAMPLES = 5


@dataclass
class StateEvent:
    """Something worth telling an operator about, with its reason.

    One entry per CONDITION, not per occurrence. `count` says how often it
    happened and `samples` holds a bounded set of distinct details, because
    the details come from the caller's own payload and can differ every time.
    """

    kind: str                       # "skip" | "unverified" | "unrecoverable"
    surface: str
    reason: str
    detail: Any = None
    count: int = 1
    samples: list = field(default_factory=list)

    def __str__(self) -> str:
        seen = f" (x{self.count})" if self.count > 1 else ""
        return f"{self.kind} {self.surface}: {self.reason} {self.detail}{seen}"


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
    #: Condition -> its event. An operator needs to know WHICH conditions
    #: occurred, not how many times: a process using a custom httpx client
    #: recorded one `client_escapes` event per LLM call, and monitor did the
    #: same with one `skip` per lossy call, in the mode designed to be left
    #: running. Keyed WITHOUT the detail, because the detail is the caller's
    #: own payload -- an agent loop appending turns reported
    #: `messages[1..N].content`, a longer tuple every call, so keying on it
    #: grew the log quadratically.
    _recorded: dict = field(default_factory=dict, repr=False, compare=False)
    #: The last observation published to `guard_health`, so a stalled call
    #: finishing late cannot republish an outage that has already recovered.
    _health_seq: int = 0
    #: Re-entrant, because a caller's own code can end up back here: `detail`
    #: is `Any`, and a value whose `__eq__` or `__repr__` records something
    #: would otherwise block forever reacquiring a plain `Lock`. The real
    #: defence is not running caller code under the lock at all -- see
    #: `_record_once` -- and this is the second line.
    _lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)
    #: True on the copy handed out by `snapshot()`. The agent's own state is
    #: never frozen.
    _frozen: bool = field(default=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"state() returns a read-only snapshot; cannot set {name!r}. "
                "The agent's account of itself is not the caller's to edit."
            )
        object.__setattr__(self, name, value)

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
        self._refuse_if_snapshot()
        self.surfaces[surface] = "unverified"
        self._record_once(StateEvent("unverified", surface, reason, detail))

    def record_guard_health(self, outcome_kind: str, sequence: int = 0) -> None:
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
        self._refuse_if_snapshot()
        with self._lock:
            if sequence and sequence < self._health_seq:
                return          # a slower, older call finishing after a newer one
            self._health_seq = max(sequence, self._health_seq)
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

    def _refuse_if_snapshot(self) -> None:
        """A write to a snapshot would silently go nowhere, which is worse
        than an error: the caller believes they recorded something."""
        if getattr(self, "_frozen", False):
            raise AttributeError(
                "state() returns a read-only snapshot; it cannot record. "
                "The agent records its own state."
            )

    def _record_once(self, event: StateEvent) -> None:
        """Append an event the first time its exact condition occurs.

        Keyed on kind, surface and reason -- NOT the detail, which the caller
        controls and which can differ on every call. Repeats increment
        `count` and contribute up to `_MAX_SAMPLES` distinct details, so an
        operator still sees examples and still learns it recurred.
        """
        self._refuse_if_snapshot()
        key = (event.kind, event.surface, event.reason)
        with self._lock:
            existing = self._recorded.get(key)
            if existing is None:
                if event.detail is not None:
                    event.samples = [event.detail]
                self._recorded[key] = event
                self.events.append(event)
                return
            existing.count += 1
            if event.detail is None or len(existing.samples) >= _MAX_SAMPLES:
                return
            known = list(existing.samples)

        # CALLER CODE MUST NOT RUN UNDER THE LOCK. `detail` is `Any`, so this
        # `==` is the caller's own `__eq__`; running it inside the critical
        # section deadlocked the thread outright when that method recorded
        # anything, and serialised all state publication when it was merely
        # slow. Reproduced before the fix with a re-entrant `__eq__`.
        if any(sample == event.detail for sample in known):
            return

        with self._lock:
            # Re-check by IDENTITY, which cannot run caller code. A racing
            # duplicate is possible and harmless; the cap still holds.
            if (len(existing.samples) < _MAX_SAMPLES
                    and not any(sample is event.detail
                                for sample in existing.samples)):
                existing.samples.append(event.detail)

    def snapshot(self) -> "State":
        """A read-only copy, which is what `state()` hands out.

        Returning the live object made the agent's own account of itself
        writable by anyone holding it: setting `lifecycle` and adding a
        surface to the returned mapping made `is_active()` report True with
        no instrumentation installed at all. An operator wiring `is_active()`
        into a health check is trusting exactly that value, and a security
        agent whose state can be edited through its public API is the
        state-lying-about-reality defect it exists to prevent, arriving
        through the front door.

        Containers are copied, so mutating the snapshot cannot reach the
        agent, and the snapshot itself refuses assignment rather than
        accepting an edit that would do nothing.
        """
        from dataclasses import replace

        with self._lock:
            copy = State(
                lifecycle=self.lifecycle,
                mode=self.mode,
                surfaces=dict(self.surfaces),
                guard_health=self.guard_health,
                events=[replace(event, samples=list(event.samples))
                        for event in self.events],
            )
        object.__setattr__(copy, "_frozen", True)
        return copy

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
