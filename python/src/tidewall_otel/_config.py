"""Configuration for the Tidewall OTel instrumentation agent.

Reads configuration from ``TIDEWALL_*`` environment variables on construction
unless explicit values are passed. The dataclass fields can also be overridden
in code if you build a config programmatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_MODES = frozenset({"enforce", "monitor", "dry-run"})
_ACTIVATION_FAILURES = frozenset({"exit", "block", "disable"})

#: Replaced by TIDEWALL_MODE. Refused rather than aliased: quietly mapping one
#: onto a mode would pick a policy the operator did not choose.
_REMOVED_VARIABLES = {
    "TIDEWALL_ON_GUARD_FAILURE": "TIDEWALL_MODE",
    "TIDEWALL_ON_UNSUPPORTED_INPUT": "TIDEWALL_MODE",
    "TIDEWALL_TIMEOUT": "TIDEWALL_SOCKET_TIMEOUT and TIDEWALL_GUARD_DEADLINE",
}


def _refuse_removed_variables() -> None:
    for name, replacement in _REMOVED_VARIABLES.items():
        if name in os.environ:
            raise ValueError(
                f"{name} was removed; use {replacement}. It is refused rather "
                f"than ignored: silently dropping a variable an operator set "
                f"deliberately leaves them believing a policy is in force."
            )


def _positive_float(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not (value > 0) or value == float("inf"):
        raise ValueError(f"{name} must be a positive finite number, got {raw!r}")
    return value



@dataclass
class TidewallConfig:
    """Runtime configuration for the Tidewall instrumentation agent.

    Environment variables:
        TIDEWALL_BASE_URL   - Tidewall guard API base URL
                              (e.g. ``http://localhost:8080``)
        TIDEWALL_TOKEN      - API token for authenticating with the guard server
        TIDEWALL_APP_ID     - Application identifier recorded in guard events
        TIDEWALL_APP_NAME   - Human-readable application name for dashboards
        TIDEWALL_USER_ID    - User identifier (defaults to ``$USER``)
        TIDEWALL_MODE       - Enforcement mode: ``enforce`` (default), ``monitor``,
                              or ``dry-run``
        TIDEWALL_LOG_LEVEL  - Logging verbosity: ``debug``, ``info`` (default),
                              ``warning``, ``error``
        TIDEWALL_SOCKET_TIMEOUT  - Per-connection read bound, seconds (default: 10)
        TIDEWALL_GUARD_DEADLINE  - Caller-latency bound, seconds (default: 10)
        TIDEWALL_ON_ACTIVATION_FAILURE
                            - ``exit`` (default), ``block`` or ``disable``

    Modes:
        ``enforce`` performs guard calls and applies the result (block / transform).
        ``monitor`` performs guard calls but only logs the result (no enforcement).
        ``dry-run`` skips guard calls entirely; useful for smoke-testing the
        instrumentation pipeline without a live backend.
    """

    base_url: str = field(
        default_factory=lambda: os.environ.get("TIDEWALL_BASE_URL", "")
    )
    token: str = field(default_factory=lambda: os.environ.get("TIDEWALL_TOKEN", ""))
    app_id: str = field(
        default_factory=lambda: os.environ.get("TIDEWALL_APP_ID", "tidewall-otel")
    )
    app_name: str = field(
        default_factory=lambda: os.environ.get(
            "TIDEWALL_APP_NAME", "Tidewall OTel Instrumentation"
        )
    )
    # NO $USER FALLBACK. The OS account name is not ours to disclose, and it
    # was previously sent twice by default -- as user_id and again as
    # extra_info.user_name. Sending an identity must be an explicit choice.
    user_id: str = field(
        default_factory=lambda: os.environ.get("TIDEWALL_USER_ID", "")
    )
    mode: str = field(default_factory=lambda: os.environ.get("TIDEWALL_MODE", "enforce"))
    log_level: str = field(
        default_factory=lambda: os.environ.get("TIDEWALL_LOG_LEVEL", "info")
    )
    # TWO BOUNDS, TWO NAMES. `socket_timeout` bounds one connection's reads;
    # `guard_deadline_s` bounds how long the CALLER waits and is enforced by
    # the BoundedExecutor. Using one name for both is how a caller/callee
    # mismatch hid through two review rounds.
    socket_timeout: float = field(
        default_factory=lambda: _positive_float("TIDEWALL_SOCKET_TIMEOUT", "10")
    )
    guard_deadline_s: float = field(
        default_factory=lambda: _positive_float("TIDEWALL_GUARD_DEADLINE", "10")
    )
    on_activation_failure: str = field(
        default_factory=lambda: os.environ.get("TIDEWALL_ON_ACTIVATION_FAILURE", "exit")
    )

    def __post_init__(self) -> None:
        """Policy errors RAISE; connection settings are collected by
        ``validate``.

        The split is deliberate. A malformed policy means the operator
        believes a rule is in force that is not, and continuing is its own
        fail-open. A missing base URL is an activation-time condition the mode
        contract already covers.
        """
        _refuse_removed_variables()

        if self.mode not in _MODES:
            raise ValueError(
                f"TIDEWALL_MODE must be one of {sorted(_MODES)}, got {self.mode!r}"
            )
        if self.on_activation_failure not in _ACTIVATION_FAILURES:
            raise ValueError(
                f"TIDEWALL_ON_ACTIVATION_FAILURE must be one of "
                f"{sorted(_ACTIVATION_FAILURES)}, got {self.on_activation_failure!r}"
            )
        if self.on_activation_failure == "block" and self.mode != "enforce":
            raise ValueError(
                f"TIDEWALL_ON_ACTIVATION_FAILURE=block installs persistent "
                f"per-call refusal, which contradicts mode {self.mode!r}, whose "
                f"runtime contract is to proceed and record. Use 'exit' or "
                f"'disable'."
            )
        if self.socket_timeout > self.guard_deadline_s:
            raise ValueError(
                f"TIDEWALL_SOCKET_TIMEOUT ({self.socket_timeout}s) exceeds "
                f"TIDEWALL_GUARD_DEADLINE ({self.guard_deadline_s}s): a socket "
                f"bound wider than the caller bound cannot be honoured"
            )

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration errors.

        An empty list means the configuration is valid. Errors are returned
        rather than raised so callers can decide whether to log-and-continue
        (fail-open) or hard-fail.
        """
        errors = []
        if not self.base_url:
            errors.append("TIDEWALL_BASE_URL must be set")
        if not self.token:
            errors.append("TIDEWALL_TOKEN must be set")
        if self.mode not in ("enforce", "monitor", "dry-run"):
            errors.append(
                f"TIDEWALL_MODE must be 'enforce', 'monitor', or 'dry-run', "
                f"got '{self.mode}'"
            )
        return errors
