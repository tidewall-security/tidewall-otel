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

    Currently only used by callers that opt into hard-fail behaviour.
    The default activation flow logs config errors and fails open instead.
    """
