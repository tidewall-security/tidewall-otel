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

    def check(
        self,
        *,
        messages: list[dict[str, str]],
        event_type: str = "input",
        model: str = "",
        llm_provider: str = "",
    ) -> GuardResult | None:
        """Send messages to the Tidewall guard for evaluation.

        Args:
            messages: Conversation messages in OpenAI Chat Completions format.
            event_type: ``input`` (default), ``output``, ``tool_input``,
                ``tool_output``, or ``tool_listing`` — controls which policy
                rules the server applies.
            model: Model identifier, recorded with the event for analytics.
            llm_provider: Provider name (``openai``, ``anthropic``...).

        Returns:
            A :class:`GuardResult` on success, or ``None`` if the call was
            skipped (dry-run) or failed (fail-open). Callers must treat
            ``None`` as "no decision" and let the original call proceed.
        """
        if self._config.mode == "dry-run":
            logger.debug(
                "[dry-run] Would guard %s (%d messages, model=%s, provider=%s)",
                event_type, len(messages), model, llm_provider,
            )
            return None

        payload: dict[str, Any] = {
            "guard_input": {"messages": messages},
            "event_type": event_type,
            "app_id": self._config.app_id,
            "user_id": self._config.user_id,
            "llm_provider": llm_provider,
            "model": model,
            "extra_info": {
                "app_name": self._config.app_name,
                "user_name": self._config.user_id,
            },
        }

        t0 = time.monotonic()
        try:
            response = post_guard(
                base_url=self._config.base_url,
                token=self._config.token,
                payload=payload,
                socket_timeout=self._config.socket_timeout,
            )
            latency = (time.monotonic() - t0) * 1000

            result_data = response.get("result") or {}
            guard_result = GuardResult(
                blocked=bool(result_data.get("blocked", False)),
                transformed=bool(result_data.get("transformed", False)),
                guard_output=result_data.get("guard_output"),
                detectors=result_data.get("detectors") or {},
                summary=str(response.get("summary") or ""),
                latency_ms=latency,
            )

            if guard_result.blocked:
                logger.warning(
                    "Tidewall BLOCKED %s (%.0fms): %s",
                    event_type, latency, guard_result.summary,
                )
            elif guard_result.has_detections:
                logger.info(
                    "Tidewall detections on %s (%.0fms): %s",
                    event_type, latency, guard_result.summary,
                )
            else:
                logger.debug(
                    "Tidewall %s clean (%.0fms)", event_type, latency
                )

            return guard_result

        except GuardAPIError:
            latency = (time.monotonic() - t0) * 1000
            logger.warning(
                "Tidewall guard call failed for %s after %.0fms — failing open",
                event_type, latency, exc_info=True,
            )
            return None
        except Exception:
            latency = (time.monotonic() - t0) * 1000
            logger.warning(
                "Unexpected error during Tidewall guard call (%s, %.0fms) — failing open",
                event_type, latency, exc_info=True,
            )
            return None
