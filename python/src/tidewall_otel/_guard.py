"""Fail-open guard client for the Tidewall API.

Wraps the minimal HTTP client from ``_http`` with the additional
behaviours expected by the wrappers:

- **Fail-open semantics**: any guard error (timeout, network, invalid
  response) is logged and treated as "no decision", so the original
  AI call always proceeds. Security must never break the host app.
- **Mode-aware behaviour**: respects ``enforce`` / ``monitor`` / ``dry-run``.
- **Structured logging**: emits one log line per guard decision so
  operators can correlate AI traffic with detector verdicts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tidewall_otel._config import TidewallConfig
from tidewall_otel._http import GuardAPIError, post_guard

logger = logging.getLogger("tidewall.otel.guard")


@dataclass
class GuardResult:
    """Parsed result from a Tidewall guard call.

    Mirrors the relevant fields of the guard API response. Additional
    fields returned by the server are silently ignored, so the agent
    keeps working when the server adds new fields.
    """

    blocked: bool = False
    transformed: bool = False
    guard_output: dict[str, Any] | None = None
    detectors: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    latency_ms: float = 0.0

    @property
    def has_detections(self) -> bool:
        """True if any detector flagged the prompt, regardless of action."""
        return any(
            isinstance(d, dict) and d.get("detected", False)
            for d in self.detectors.values()
        )


class TidewallGuard:
    """Calls the Tidewall guard API with fail-open semantics.

    The class encapsulates everything the SDK wrappers need to know about
    the guard backend, so the wrappers themselves stay focused on
    SDK-specific concerns (message normalization, span creation, etc.).

    Behaviour by mode:
      * ``dry-run``  — :meth:`check` returns ``None`` immediately. No HTTP.
      * ``monitor``  — guard call is made; result is logged but never enforced.
      * ``enforce``  — guard call is made and the result drives blocking
        / transformation decisions in the calling wrapper.
    """

    def __init__(self, config: TidewallConfig) -> None:
        self._config = config

    def _payload_for(
        self,
        guard_input: dict,
        *,
        event_type: str = "input",
        model: str = "",
        llm_provider: str = "",
    ) -> dict:
        """Wrap a guard input in the request envelope.

        Task 3 owns ``post_guard`` and its ``payload``; this builds that
        payload, so the envelope has one definition rather than being
        assembled inline wherever a request is sent.
        """
        extra_info: dict[str, Any] = {"app_name": self._config.app_name}
        payload: dict[str, Any] = {
            "guard_input": guard_input,
            "event_type": event_type,
            "app_id": self._config.app_id,
            "llm_provider": llm_provider,
            "model": model,
            "extra_info": extra_info,
        }

        # OMITTED, not blank. An empty string still asserts a field the
        # operator never set, and a downstream consumer treating presence as
        # meaningful would record it as an identity.
        if self._config.user_id:
            payload["user_id"] = self._config.user_id
            extra_info["user_name"] = self._config.user_id

        return payload

    def check_raw(self, *, guard_input: dict, event_type: str = "input",
                  model: str = "", llm_provider: str = "") -> dict:
        """Perform the request and return the DECODED BODY.

        Raises GuardUnreachable / GuardTimeout / GuardSchemaInvalid rather
        than swallowing them into None the way :meth:`check` does. Anything
        else propagates and dispatch files it as invariant_violated.

        The transport raises those types directly (they subclass
        GuardAPIError), so there is no seam here to sniff causes at.
        """
        return post_guard(
            base_url=self._config.base_url,
            token=self._config.token,
            payload=self._payload_for(guard_input, event_type=event_type,
                                      model=model, llm_provider=llm_provider),
            socket_timeout=self._config.socket_timeout,
        )

