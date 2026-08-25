"""Custom exceptions raised by the Tidewall OTel instrumentation agent."""

from __future__ import annotations

from typing import Any


class TidewallError(Exception):
    """Base exception for Tidewall OTel instrumentation errors."""


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
        super().__init__(f"Tidewall blocked request: {summary}")


class TidewallConfigError(TidewallError):
    """Raised when configuration is invalid or incomplete.

    Raised BY DEFAULT: ``TIDEWALL_ON_ACTIVATION_FAILURE`` defaults to ``exit``,
    so invalid configuration stops the process rather than letting it continue
    believing it is guarded. The other policies are ``disable`` (run unguarded,
    with `state()` saying so) and ``block`` (install refusers, so calls fail
    rather than pass unchecked).

    This docstring previously said the default "logs config errors and fails
    open instead" -- describing the exact behaviour this programme removed.
    """


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
