"""Custom exceptions raised by the Tidewall OTel instrumentation agent."""

from __future__ import annotations

from typing import Any


class TidewallError(Exception):
    """Base exception for Tidewall OTel instrumentation errors.

    Every declination carries an ``outcome_kind``. The README documents one
    vocabulary and tells operators to catch this class, but only
    `TidewallRefusedError` defined the attribute -- so code that branched on
    it broke on `blocked`, the ordinary policy path this product exists to
    produce, and the most likely branch anyone writes.
    """

    #: Overridden per subclass and per raise site. A closed vocabulary:
    #: `blocked`, `lossy`, `mutated_during_guard`, `unverifiable_payload`,
    #: `unreachable`, `timeout`, `saturated`, `schema_invalid`,
    #: `invariant_violated`, `config_invalid`.
    outcome_kind: str = "invariant_violated"


class TidewallBlockedError(TidewallError):
    """Raised when a Tidewall policy blocks an AI request in enforce mode.

    Caught by application code that wants to detect blocks and surface a
    user-friendly message. In ``monitor`` and ``dry-run`` modes this exception
    is never raised — the original AI call always proceeds.

    Attributes:
        summary: Human-readable description from the guard response.
        detectors: Dict of detector results that triggered the block.
    """

    def __init__(
        self, summary: str, detectors: dict[str, Any] | None = None
    ) -> None:
        self.summary = summary
        self.detectors = detectors or {}
        self.outcome_kind = "blocked"
        super().__init__(f"Tidewall blocked request: {summary}")


class TidewallConfigError(TidewallError):
    """Raised when configuration is invalid or incomplete.

    Its ``outcome_kind`` is ``config_invalid``.

    Raised BY DEFAULT: ``TIDEWALL_ON_ACTIVATION_FAILURE`` defaults to ``exit``,
    so invalid configuration stops the process rather than letting it continue
    believing it is guarded. The other policies are ``disable`` (run unguarded,
    with `state()` saying so) and ``block`` (install refusers, so calls fail
    rather than pass unchecked). There is no fail-open default: a process that
    logs a configuration error and continues is unguarded while believing
    otherwise.
    """

    outcome_kind = "config_invalid"


class TidewallRefusedError(TidewallError):
    """A call dispatch refused: guard failure, or input it cannot represent.

    Subclasses the EXISTING TidewallError from ``_exceptions`` rather than
    introducing a second base. A caller wants one ``except`` clause covering
    every way Tidewall can decline a call -- a blocked verdict and a refused
    one are the same event to the application -- and two unrelated hierarchies
    would silently let one escape a handler written for the other.
    """

    def __init__(self, message: str, outcome_kind: str = "") -> None:
        super().__init__(message)
        self.outcome_kind = outcome_kind


class LossyInputError(TidewallRefusedError):
    """The call carries content the guard cannot be shown faithfully."""

    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__(
            f"refusing: input cannot be represented to the guard at {list(paths)}",
            outcome_kind="lossy",
        )
        self.paths = paths

    #: `paths` names WHICH arguments could not be represented, because the
    #: caller can act on that: dropping `extra_body` makes the call
    #: inspectable, whereas a guard failure is not something they can fix.
    #: Public for the same reason -- a caller who cannot name the type cannot
    #: distinguish the two without importing a private module.
