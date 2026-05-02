"""Configuration for the Tidewall OTel instrumentation agent.

Reads configuration from ``TIDEWALL_*`` environment variables on construction
unless explicit values are passed. The dataclass fields can also be overridden
in code if you build a config programmatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
        TIDEWALL_TIMEOUT    - Per-request timeout in seconds (default: 10)

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
    user_id: str = field(
        default_factory=lambda: os.environ.get(
            "TIDEWALL_USER_ID", os.environ.get("USER", "unknown")
        )
    )
    mode: str = field(default_factory=lambda: os.environ.get("TIDEWALL_MODE", "enforce"))
    log_level: str = field(
        default_factory=lambda: os.environ.get("TIDEWALL_LOG_LEVEL", "info")
    )
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("TIDEWALL_TIMEOUT", "10"))
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
